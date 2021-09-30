import os
import json
import logging
import datetime
from decimal import Decimal
from typing import Tuple, List, Dict, TypeVar, Set, Optional, Any
import requests
import pandas as pd



logging.basicConfig()
log = logging.getLogger(__name__[:-3])
log.setLevel(logging.INFO)

T = TypeVar('T') 

gauge_data_uri = os.path.join(os.path.dirname(os.path.abspath(__file__)), \
                              'gauge_data/bom_gauge_data.csv')
gauges: pd.core.frame.DataFrame = None

STATE_URLS = {
    'NSW': 'realtimedata.waternsw.com.au',
    'QLD': 'water-monitoring.information.qld.gov.au',
    'VIC': 'data.water.vic.gov.au'
}


STATE_LEVEL_VarFrom = {
    'NSW': Decimal('130.00'),
    'VIC' : Decimal('100.00'),
    'QLD' : Decimal('100.00')
}

STATE_LEVEL_VarTo = {
    'NSW': Decimal('130.00'),
    'VIC' : Decimal('100.00'),
    'QLD' : Decimal('100.00')
}

STATE_FLOW_VarFrom = {
    'NSW': Decimal('100.00'),
    'VIC' : Decimal('100.00'),
    'QLD' : Decimal('100.00')
}

STATE_FLOW_VarTo = {
    'NSW': Decimal('141.00'),
    'VIC' : Decimal('141.00'),
    'QLD' : Decimal('141.00')
}

MAX_SITES_PER_REQUEST = {
    'NSW': 10,
    'VIC': 10,
    'QLD': 10 
}


def init() -> None:
    '''
    Loads gauges from disk. This will dynamically trigger when other libraries require
    gague data.
    '''
    global gauges
    gauges = pd.read_csv(gauge_data_uri, skiprows=1, skipfooter=1,
                         names=['gauge_name', 'gauge_number',
                                'gauge_owner', 'lat', 'long'], engine='python')
    gauges['State'] = gauges['gauge_owner'].str.strip().str.split(' ', 1).str[0]
    gauges = gauges.drop(['lat', 'long', 'gauge_owner'], axis=1)


def get_states_for_gauge(gauge_number: str) -> Set[str]:
    '''
    Given a gauge number, returns a set of states which that gauge may belong to.
    '''

    # TODO-Review There's possibly a bug here - some gauges, such as 422204A (CULGOA @ WHYENBAH)
    # Return two results in different states, that may create integrity issues and
    # should be investigated. It is caused by the data within bom_gauge_data.csv. not application
    # logic.
    if not hasattr(gauges, "empty") or gauges.empty:
        init()
    matching = list(gauges[gauges['gauge_number'] == gauge_number]['State'])
    if len(matching) > 1:
        log.warning(f'Gauge {gauge_number} has many state results: {matching}')
    return set(matching)


def sort_gauges_by_state(gauge_numbers: List[str]) -> Dict[str, List[str]]:
    '''
    Splits the listed gauges into state-based lists.
    '''
    states: Dict[str, List[str]] = {
        'NSW': [],
        'QLD': [],
        'VIC': [],
        'rest': [],
    }
    for gauge in gauge_numbers:
        for gauge_state in get_states_for_gauge(gauge):
            if gauge_state not in states:
                gauge_state = 'rest'
            if gauge in states[gauge_state]:
                continue
            states[gauge_state].append(gauge)
    return states


def call_state_api(state: str, indicative_sites: List[str], start_time: datetime.date,
                   end_time: datetime.date, data_source: str, var: str,
                   interval: str, data_type: str) -> Dict[str, Any]:
    '''
    Sends a web request with a destination based on `state`.

    Returns a JSON dict object containing web responses, and will fail if the server returns
    either a non HTTP-200 error code, or invalid JSON
    '''
    if not isinstance(start_time, datetime.date):
        raise TypeError('start_time must be a datetime.date object, but got type '
                        f'{type(start_time)} (value: \'{start_time}\')')
    if not isinstance(end_time, datetime.date):
        raise TypeError('end_time must be a datetime.date object, but got type '
                        f'{type(end_time)} (value: \'{end_time}\')')
    
    url = STATE_URLS[state]

    if (var=="L"):
        var_from = STATE_LEVEL_VarFrom[state]
        var_to = STATE_LEVEL_VarTo[state]
    elif (var=="F"):
        var_from = STATE_FLOW_VarFrom[state]
        var_to = STATE_FLOW_VarTo[state]

    sites = ','.join(indicative_sites)
    data = {
        'params': {
            'site_list': sites,
            'start_time': start_time.strftime('%Y%m%d') + '000000',
            'varfrom': f'{var_from:.2f}',
            'interval': str(interval),
            'varto': f'{var_to:.2f}', 
            'datasource': str(data_source),
            'end_time': end_time.strftime('%Y%m%d') + '000000',
            'data_type': str(data_type),
            'multiplier':'1'
        },
        'function':'get_ts_traces',
        'version':'2'
    }
    json_data = json.dumps(data, separators=(',', ':'))

 
    req_url = f'https://{url}/cgi/webservice.pl?{json_data}'
    req_url = req_url.replace(' ', '%20')

    # TODO-idiosyncratic the use of JSON in the query string seems werid, this should be a HTTP POST
    # but requires endpoints to support it..
    log.debug(f'ending request to URL \'{req_url}\'')
    r = requests.get(req_url)
    if not r.status_code == 200: 
        raise requests.HTTPError(f'Request to \'{url}\' failed with HTTP Response code '
                                 f'{r.status_code} and HTTP Response:\n{r.content}')
    try:
        return json.loads(r.content)
    except json.decoder.JSONDecodeError:
        raise json.decoder.JSONDecodeError(
            f'Unable to parse response to request to \'{url}\'. The server returned invalid JSON '
            f' data. Got HTTP Response code {r.status_code} and HTTP Response:\n{r.content}',
            r.content.decode(), 0)


def extract_data(state: str, data) -> List[List[Any]]:
    """
    Collects 
    """
    
    try:
        extracted = []
        for sample in data['_return']['traces']:
            for obs in sample['trace']:
                # TODO-Detail - put detail re the purpose of obs['q'] - I don't know what/why this
                # logic exists, it's obviously to sanitise data but unclear on what/why
                # TODO-idiosyncratic: was < 999 prior to refactor, this means that 998 is the max
                # accepted number, this would presumably be 999, but I can't say for sure
                if int(obs['q']) >= 999:
                    continue    
                obsdate = datetime.datetime.strptime(str(obs['t']), '%Y%m%d%H%M%S').date()
                objRow = [state, sample['site'], 'WATER', obsdate, obs['v'], obs['q']]
                extracted.append(objRow)
    except KeyError:
        extracted = []
    return extracted


def split_into_chunks(input_list: List[T], maxlen: int) -> List[List[T]]:
    '''
    Splits a list into many lists of maximum `maxlen` length. Let input_list = [1,2,3,4].
    When:

    - `maxlen=1` => `[[1],[2],[3],[4]]`
    - `maxlen=2` => `[[1, 2],[3, 4]]`
    - `maxlen=3` => `[[1, 2, 3], [4]]`
    - `maxlen=4` => `[[1, 2, 3, 4]]`
    - `maxlen=5` => `[[1, 2, 3, 4]]`
    '''

    assert(maxlen) > 0
    l = []
    for i in range(0, len(input_list), maxlen):
        l.append(input_list[i:i + maxlen])
    return l


def process_gauge_pull(sitelist: List[str], callstate: str, call_data_source: str,
                       start_time_user: datetime.date, end_time_user: datetime.date,
                       var: str, interval: str,
                       data_type: str) -> List[List[Any]]:
    '''
    Intermediate function which splits many gauge_pull records into separate web requests
    and provides user feedback on progress
    '''
    max_sites_per_request = MAX_SITES_PER_REQUEST[callstate]
    site_chunks = split_into_chunks(sitelist, max_sites_per_request)
    response_data: List[List[str]] = []
    for index, s in enumerate(site_chunks):
        log.info(f'{callstate} - Request {index+1} of {len(site_chunks)}')
        ret = call_state_api(callstate, s, start_time_user, end_time_user,
                             call_data_source, var, interval, data_type)
        response_data += extract_data(callstate, ret)
    return response_data


def gauge_pull(gauge_numbers: List[str], start_time_user: datetime.date, end_time_user: datetime.date,
               var: str = 'F',
               interval: str = 'day', data_type: str = 'mean') -> pd.DataFrame:
    '''
    Given a list of gauge numbers, sorts the list into state groups, and queries relevant
    HTTP endpoints for data, returning as a Pandas dataframe object.
    '''


    gauges_by_state = sort_gauges_by_state(gauge_numbers)
    data: List[List[List[Any]]] = []
    data += process_gauge_pull(gauges_by_state['NSW'], 'NSW', 'CP', start_time_user,
                               end_time_user, var, interval, data_type)
    data += process_gauge_pull(gauges_by_state['VIC'], 'VIC', 'PUBLISH', start_time_user,
                               end_time_user, var, interval, data_type)
    data += process_gauge_pull(gauges_by_state['QLD'], 'QLD', 'AT', start_time_user,
                               end_time_user, var, interval, data_type)
    cols = ['DATASOURCEID', 'SITEID', 'SUBJECTID', 'DATETIME', 'VALUE', 'QUALITYCODE']
    flow_data_frame = pd.DataFrame(data=data, columns=cols)
    return flow_data_frame
