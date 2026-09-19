import matplotlib.pyplot as plt

# Results from inner and outer product MMs from main.cpp
N = [16, 32, 64, 128, 256, 512, 1024]
inner_ns = [3600, 21950, 234208, 2059421, 14703833, 112376870, 1074083124]
outer_ns = [1525, 6975, 48441, 238917, 1337569, 12037410, 91357286]

plt.plot(N, inner_ns, marker="o", label="Inner product")
plt.plot(N, outer_ns, marker="s", label="Outer product")

# Log-log axes since N doubles each step and run time spans ~6 orders of magnitude
plt.xscale("log", base=2)
plt.yscale("log")
plt.xticks(N, [str(n) for n in N])

plt.xlabel("Matrix dimension N")
plt.ylabel("Average run time over 5 runs (ns)")
plt.title("Inner vs. outer product MMM run time")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig("plot.png", dpi=200, bbox_inches="tight")
