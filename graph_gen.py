# %%
import csv
import matplotlib.pyplot as plt
from statistics import variance
from math import sqrt
import os


# %%
RR_DATA_FILE = 'sim_data_rr.csv'
FIFO_DATA_FILE = 'sim_data_fifo.csv'
GRAPHS_DIR = 'graphs'

NUM_RUNS = 10

numUsersRR = []
responseTimesRR = []
goodPutsRR = []
badPutsRR = []
requestDropRatesRR = []
coreUtilisationsRR = []
avgQueueLengthRR = []
avgWaitingTimeInQueueRR = []

numUsersFIFO = []
responseTimesFIFO = []
goodPutsFIFO = []
badPutsFIFO = []
requestDropRatesFIFO = []
coreUtilisationsFIFO = []
avgQueueLengthFIFO = []
avgWaitingTimeInQueueFIFO = []


# %%
def load_data(filename: str, numUsers: list, responseTimes: list, goodPuts: list, badPuts: list, requestDropRates: list, coreUtilisations: list, avgQueueLength: list, avgWaitingTimeInQueue: list):
    with open(filename) as f:
        reader = csv.reader(f)

        next(reader)
        next(reader)
        next(reader)

        for line in reader:
            numUsers.append(int(line[0]))
            responseTimes.append(list(map(float,line[1:NUM_RUNS+1])))
            goodPuts.append(list(map(float,line[(NUM_RUNS + 1):(2 * NUM_RUNS) + 1])))
            badPuts.append(list(map(float,line[(2 * NUM_RUNS + 1):(3 * NUM_RUNS) + 1])))
            requestDropRates.append(list(map(float,line[(3 * NUM_RUNS + 1):(4 * NUM_RUNS) + 1])))
            coreUtilisations.append(list(map(float,line[(4 * NUM_RUNS + 1):(5 * NUM_RUNS) + 1])))
            avgQueueLength.append(list(map(float,line[(5 * NUM_RUNS + 1):(6 * NUM_RUNS) + 1])))
            avgWaitingTimeInQueue.append(list(map(float,line[(6 * NUM_RUNS + 1):(7 * NUM_RUNS) + 1])))

def avg(l: list) -> float:
    return sum(l) / len(l)


# %%
# create directory for graphs
if not os.path.exists(GRAPHS_DIR):
    os.makedirs(GRAPHS_DIR)

# %%
load_data(RR_DATA_FILE, numUsersRR, responseTimesRR, goodPutsRR, badPutsRR, requestDropRatesRR, coreUtilisationsRR, avgQueueLengthRR, avgWaitingTimeInQueueRR)
load_data(FIFO_DATA_FILE, numUsersFIFO, responseTimesFIFO, goodPutsFIFO, badPutsFIFO, requestDropRatesFIFO, coreUtilisationsFIFO, avgQueueLengthFIFO, avgWaitingTimeInQueueFIFO)


# %%
plt.clf()
plt.plot(numUsersRR, list(map(avg, goodPutsRR)), label='Round Robin')
plt.plot(numUsersFIFO, list(map(avg, goodPutsFIFO)), label='FIFO')
plt.xlabel('Number of users')
plt.ylabel('Goodput (reqs/s)')
plt.title('Goodput vs Number of Users (M)')
plt.minorticks_on()
plt.grid(True, 'both')
plt.legend()
plt.savefig('graphs/goodput.png', bbox_inches='tight')
# plt.show()

# %%
plt.clf()
plt.plot(numUsersRR, list(map(avg, badPutsRR)), label='Round Robin')
plt.plot(numUsersFIFO, list(map(avg, badPutsFIFO)), label='FIFO')
plt.xlabel('Number of users')
plt.ylabel('Badput (reqs/s)')
plt.title('Badput vs Number of Users (M)')
plt.minorticks_on()
plt.grid(True, 'both')
plt.legend()
plt.savefig('graphs/badput.png', bbox_inches='tight')
# plt.show()

# %%
plt.clf()
plt.plot(numUsersRR, [ sum(x)  for x in zip( map(avg, goodPutsRR) , map(avg, badPutsRR)) ], label='Round Robin')
plt.plot(numUsersFIFO, [ sum(x)  for x in zip( map(avg, goodPutsFIFO) , map(avg, badPutsFIFO)) ], label='FIFO')
plt.xlabel('Number of users')
plt.ylabel('Throughput (reqs/s)')
plt.title('Throughput (Λ) vs Number of Users (M)')
plt.minorticks_on()
plt.grid(True, 'both')
plt.legend()
plt.savefig('graphs/throughput.png', bbox_inches='tight')
# plt.show()

# %%
plt.clf()
plt.plot(numUsersRR, list(map(avg, requestDropRatesRR)), label='Round Robin')
plt.plot(numUsersFIFO, list(map(avg, requestDropRatesFIFO)), label='FIFO')
plt.xlabel('Number of users')
plt.ylabel('Request Drop Rate (reqs/s)')
plt.title('Request Drop Rate vs Number of Users (M)')
plt.minorticks_on()
plt.grid(True, 'both')
plt.legend()
# plt.show()
plt.savefig('graphs/req_drop_rate.png', bbox_inches='tight')

# %%
plt.clf()
plt.plot(numUsersRR, list(map(avg, coreUtilisationsRR)), label='Round Robin')
plt.plot(numUsersFIFO, list(map(avg, coreUtilisationsFIFO)), label='FIFO')
plt.xlabel('Number of users')
plt.ylabel('CPU Utilisation')
plt.title('CPU Utilisation (ρ) vs Number of Users (M)')
plt.minorticks_on()
plt.grid(True, 'both')
plt.legend()
plt.savefig('graphs/utilisation.png', bbox_inches='tight')
# plt.show()

# %%
plt.clf()
plt.plot(numUsersRR, list(map(avg, avgQueueLengthRR)), label='Round Robin')
plt.plot(numUsersFIFO, list(map(avg, avgQueueLengthFIFO)), label='FIFO')
plt.xlabel('Number of users')
plt.ylabel('Average Queue Length')
plt.title('Average Queue Length vs Number of Users (M)')
plt.minorticks_on()
plt.grid(True, 'both')
plt.legend()
plt.savefig('graphs/avg_queue_len.png', bbox_inches='tight')
# plt.show()

# %%
plt.clf()
plt.plot(numUsersRR, list(map(avg, avgWaitingTimeInQueueRR)), label='Round Robin')
plt.plot(numUsersFIFO, list(map(avg, avgWaitingTimeInQueueFIFO)), label='FIFO')
plt.xlabel('Number of users')
plt.ylabel('Average Waiting Time in Queue (s)')
plt.title('Average Waiting Time in Queue vs Number of Users (M)')
plt.minorticks_on()
plt.grid(True, 'both')
plt.legend()
plt.savefig('graphs/avg_queue_wait_time.png', bbox_inches='tight')
# plt.show()

# %%
def plot_confidence_interval(x: int, values: list[float], z: float, color: str) -> None:

    mean = avg(values)
    fac = sqrt(variance(values)/len(values))

    top = mean + z * fac
    bottom = mean - z * fac

    plt.plot([x, x], [ bottom, top ], color=color)

    


# %%
plt.clf()
for i in range(len(numUsersRR)):
    plot_confidence_interval(numUsersRR[i], responseTimesRR[i], 1.96, 'tab:blue')

plt.xlabel('Number of users')
plt.ylabel('Response Time (s)')
plt.title('Response Time Confidence Interval (90%) vs Number of Users (M) [RoundRobin]')
plt.minorticks_on()
plt.grid(True, 'both')
plt.savefig('graphs/resp_time_ci_rr.png', bbox_inches='tight')
# plt.show()

# %%
plt.clf()
for i in range(min(15, len(numUsersRR))):
    plot_confidence_interval(numUsersRR[i], responseTimesRR[i], 1.96, 'tab:blue')

plt.xlabel('Number of users')
plt.ylabel('Response Time (s)')
plt.title('Response Time Confidence Interval (90%) vs Number of Users (M) [RoundRobin]')
plt.minorticks_on()
plt.grid(True, 'both')
plt.savefig('graphs/resp_time_ci_rr_zoomed.png', bbox_inches='tight')
# plt.show()

# %%
plt.clf()
for i in range(len(numUsersFIFO)):
    plot_confidence_interval(numUsersFIFO[i], responseTimesFIFO[i], 1.96, 'tab:orange')

plt.xlabel('Number of users')
plt.ylabel('Response Time (s)')
plt.title('Response Time Confidence Interval (90%) vs Number of Users (M) [FIFO]')
plt.minorticks_on()
plt.grid(True, 'both')
plt.savefig('graphs/resp_time_ci_fifo.png', bbox_inches='tight')
# plt.show()

# %%
plt.clf()
for i in range(min(15,len(numUsersFIFO))):
    plot_confidence_interval(numUsersFIFO[i], responseTimesFIFO[i], 1.96, 'tab:orange')

plt.xlabel('Number of users')
plt.ylabel('Response Time (s)')
plt.title('Response Time Confidence Interval (90%) vs Number of Users (M) [FIFO]')
plt.minorticks_on()
plt.grid(True, 'both')
plt.savefig('graphs/resp_time_ci_fifo_zoomed.png', bbox_inches='tight')
# plt.show()

# %%
plt.clf()
plt.plot(numUsersRR, list(map(avg, responseTimesRR)), label='Round Robin')
plt.plot(numUsersRR, list(map(avg, responseTimesFIFO)), label='FIFO')
plt.xlabel('Number of users')
plt.ylabel('Response Time (s)')
plt.title('Response Time (s) vs Number of Users (M)')
plt.minorticks_on()
plt.grid(True, 'both')
plt.legend()
plt.savefig('graphs/resp_time.png', bbox_inches='tight')
# plt.show()

# %%
exp_users = [ 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240 ]
exp_throughputs = [1.375, 2.75, 4.125, 5.5, 6.875, 7.983333333, 9.283333333, 10.53333333, 11.35833333, 11.35, 11.34166667, 11.35833333, 11.35, 11.34166667, 11.35, 11.34166667 ]
exp_resp_time = [0.28, 0.36, 0.44, 0.51, 0.56, 0.66, 0.71, 0.77, 1.15, 2.46, 3.52, 4.72, 5.9, 7.02, 8.09, 9.2 ]
exp_util = [ 0.1036, 0.2038, 0.3005, 0.413, 0.5364, 0.6267, 0.7708, 0.915, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ]

mva_users = [ 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240 ]
mva_resp_time = [0.08989, 0.10354000000000001, 0.12190999999999999, 0.14781, 0.18668, 0.25001, 0.36501999999999996, 0.6009099999999999, 1.1123800000000001, 2.03794, 3.20123, 4.40001, 5.6, 6.8, 8.0, 9.2]
mva_throughputs = [1.5, 3.0, 4.4, 5.8999999999999995, 7.4, 8.8, 10.1, 11.299999999999999, 12.1, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5]
mva_util = [0.119, 0.238, 0.356, 0.473, 0.589, 0.702, 0.81, 0.906, 0.972, 0.997, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


# %%
plt.clf()
plt.plot(numUsersRR, list(map(avg, responseTimesRR)), label='Simulator RR')
plt.plot(numUsersFIFO, list(map(avg, responseTimesFIFO)), label='Simulator FIFO')
plt.plot(exp_users,exp_resp_time, label='Experiment')
plt.plot(mva_users, mva_resp_time, label='MVA')
plt.xlabel('Number of users')
plt.ylabel('Response Time (s)')
plt.title('Response Time (s) vs Number of Users (M)')
plt.minorticks_on()
plt.grid(True, 'both')
plt.legend()
plt.savefig('graphs/resp_time_sim_exp_mva.png')
# plt.show()

# %%
plt.clf()
plt.plot(numUsersRR, list(map(avg, coreUtilisationsRR)), label='Simulator RR')
plt.plot(numUsersFIFO, list(map(avg, coreUtilisationsFIFO)), label='Simulator FIFO')
plt.plot(exp_users, exp_util, label='Experiment')
plt.plot(mva_users, mva_util, label='MVA')
plt.xlabel('Number of users')
plt.ylabel('Utilisation)')
plt.title('Utilisation (ρ) vs Number of Users (M)')
plt.minorticks_on()
plt.grid(True, 'both')
plt.legend()
plt.savefig('graphs/utilisation_sim_exp_mva.png')
# plt.show()

# %%
plt.clf()
plt.plot(numUsersRR, [ sum(x)  for x in zip( map(avg, goodPutsRR) , map(avg, badPutsRR)) ], label='Simulator RR')
plt.plot(numUsersFIFO, [ sum(x)  for x in zip( map(avg, goodPutsFIFO) , map(avg, badPutsFIFO)) ], label='Simulator FIFO')
plt.plot(exp_users, exp_throughputs, label='Experiment')
plt.plot(mva_users, mva_throughputs, label='MVA')
plt.xlabel('Number of users')
plt.ylabel('Throughput (req/s)')
plt.title('Throughput (Λ) vs Number of Users (M)')
plt.minorticks_on()
plt.grid(True, 'both')
plt.legend()
plt.savefig('graphs/throughput_sim_exp_mva.png')
# plt.show()

# %%
print(f'Graphs generated under {GRAPHS_DIR} directory')


