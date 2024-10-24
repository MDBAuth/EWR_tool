import pstats, cProfile


import run_EWR
 
cProfile.runctx("run_EWR.main()", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats(50)