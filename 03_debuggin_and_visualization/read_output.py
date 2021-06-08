import pstats

# output.txt cannot be read directly
p = pstats.Stats('output.txt')
p.sort_stats('cumulative').print_stats(100)