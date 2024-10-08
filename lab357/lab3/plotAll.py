import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})

fig, ax = plt.subplots(figsize=(15, 10))
x = [8, 32, 256, 512, 1024, 4096, 8192]
gflops_naive  = [0.2, 0.5, 0.6, 0.7, 0.7, 0.6, 0.5]
gflops_cblas  = [0.1, 2.1, 0.5, 23.8, 39.7, 51.3, 90.1]
gflops_openmp = [0.02, 3.3, 3.8, 3.4, 3.6, 3.8, 2.7]
gflops_thread = [0.15, 0.4, 0.55, 0.65, 0.75, 0.7, 0.6]

plt.plot(x, gflops_naive, label='naive', color='blue', linestyle='-', linewidth=3.0)
plt.plot(x, gflops_thread, label='thread', color='yellow', linestyle=':', linewidth=3.0)
plt.plot(x, gflops_cblas, label='cblas', color='green', linestyle='-.', linewidth=3.0)
plt.plot(x, gflops_openmp, label='openmp', color='purple', linestyle=':', linewidth=3.0)

plt.title('dgemm')
plt.xlabel('matrix size')
plt.ylabel('gflops')
plt.legend()

# plt.savefig('result.png')
plt.show()
