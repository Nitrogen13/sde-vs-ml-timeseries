def save_plot(plot, fname, q):

    plot.xlabel("t")
    plot.ylabel("S")
    plot.savefig(fname + "_" + "_".join(map(lambda k: f"{k[0]}_{k[1]}", q.items())) + ".png", dpi=None, facecolor='w', edgecolor='w',
                 orientation='portrait', papertype=None, format=None,
                 transparent=False, bbox_inches=None, pad_inches=0.1,
                 frameon=None, metadata=None)
