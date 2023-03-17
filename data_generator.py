from simulator import simulator, SchedulerType
from multiprocessing import Process
import csv

NUM_USERS = [ 1, 2, 3, 4, 5, 10 ] + list(range(15, 301, 15)) + list(range(350, 600, 50))
CLOCKS_PER_SEC = 1000000
MAX_THREADS = 10
NUM_CPUS = 2
THINK_TIME = 10000000
AVG_SERVICE_TIME = 80000
QUANTA_BURST_TIME = 20000
AVG_INTERARRIVAL_TIME = 1000000
TIMEOUT = 2000000
SIMULATION_TIME = 300000000
CTX_SWITCH_OVERHEAD = 1000
MAX_QUEUE_SIZE = 200
RETRY_PROB = 0.5
RETRY_DELAY = 5000

OUTPUT_DATA_FILE_RR = "sim_data_rr.csv"
OUTPUT_DATA_FILE_FIFO = "sim_data_fifo.csv"

NUM_RUNS = 10

SEEDS = [
    1,
 1973272912, 281629770,  20006270,1280689831,2096730329,1933576050,
  913566091, 246780520,1363774876, 604901985,1511192140,1259851944,
]


CSV_COLUMNS = [ "NUM_USERS" ] + [ f"RESPONSE_TIMES_RUN_{i+1}" for i in range(NUM_RUNS) ] + \
              [ f"GOODPUT_RUN_{i+1}" for i in range(NUM_RUNS) ] +\
              [ f"BADPUT_RUN_{i+1}" for i in range(NUM_RUNS) ] +\
              [ f"DROP_RATE_RUN_{i+1}" for i in range(NUM_RUNS) ] +\
              [ f"UTIL_RUN_{i+1}" for i in range(NUM_RUNS) ] +\
              [ f"AVG_QUEUE_LEN_RUN_{i+1}" for i in range(NUM_RUNS) ] +\
              [ f"QUEUE_WAIT_RUN_{i+1}" for i in range(NUM_RUNS) ]
              
def print_run_config(f, scheduler: SchedulerType):
    f.writerow([
        'CLOCKS_PER_SEC', str(CLOCKS_PER_SEC),
        'MAX_THREADS',  str(MAX_THREADS),
        'NUM_CPUS', str(NUM_CPUS),
        'THINK_TIME',  str(THINK_TIME),
        'AVG_SERVICE_TIME', str(AVG_SERVICE_TIME),
        'QUANTA_BURST_TIME', str(QUANTA_BURST_TIME),
        'AVG_INTERARRIVAL_TIME', str(AVG_INTERARRIVAL_TIME)
    ])

    f.writerow([
        'TIMEOUT', str(TIMEOUT),
        'SIMULATION_TIME',  str(SIMULATION_TIME),
        'CTX_SWITCH_OVERHEAD', str(CTX_SWITCH_OVERHEAD),
        'MAX_QUEUE_SIZE',  str(MAX_QUEUE_SIZE),
        'RETRY_PROB', str(RETRY_PROB),
        'RETRY_DELAY', str(RETRY_DELAY),
        'SCHEDULER', scheduler.name
    ])

def write_data_row(f, n_users, responseTimes, goodPuts, badPuts, requestDropRates, coreUtilsations, numRetriesPerTask, avgWaitingTimeInQueue):
    f.writerow( [str(n_users)] + list(map(str, responseTimes)) + list(map(str, goodPuts)) + list(map(str, badPuts)) + list(map(str, requestDropRates)) + list(map(str, coreUtilsations)) + list(map(str, numRetriesPerTask)) + list(map(str, avgWaitingTimeInQueue)) )


def generate_data(output_file_name: str, schedulerType: SchedulerType):
    file = open(output_file_name, "w", newline='')
    writer = csv.writer(file)

    print_run_config(writer, schedulerType)
    writer.writerow(CSV_COLUMNS)
    for n_user in NUM_USERS:        
        responseTimes = []
        goodPuts = []
        badPuts = []
        requestDropRates = []
        coreUtilsations = []
        numRetriesPerTask = []
        avgWaitingTimeInQueue = []

        for i in range(NUM_RUNS):
            sim = simulator(NUM_CPUS, MAX_THREADS)

            sim.run_simulation(n_user, THINK_TIME, AVG_SERVICE_TIME, QUANTA_BURST_TIME, AVG_INTERARRIVAL_TIME, TIMEOUT, SIMULATION_TIME, CTX_SWITCH_OVERHEAD, MAX_QUEUE_SIZE, RETRY_PROB, RETRY_DELAY, SchedulerType.ROUNDROBIN, SEEDS[i], i)

            responseTimes.append(sim.getAvgResponseTime() / CLOCKS_PER_SEC)
            goodPuts.append(sim.getGoodPut() * CLOCKS_PER_SEC)
            badPuts.append(sim.getBadPut() * CLOCKS_PER_SEC)
            requestDropRates.append(sim.getRequestDropRate() * CLOCKS_PER_SEC)
            coreUtilsations.append(sim.getCoreUtilisation())
            numRetriesPerTask.append(sim.getAvgQueueLength())
            avgWaitingTimeInQueue.append(sim.getAvgWaitingTimeInQueue() / CLOCKS_PER_SEC)
        write_data_row(writer, n_user, responseTimes, goodPuts, badPuts, requestDropRates, coreUtilsations, numRetriesPerTask, avgWaitingTimeInQueue)       
    file.close()


def main():
    rr_p  = Process(target=generate_data, args=(OUTPUT_DATA_FILE_RR, SchedulerType.ROUNDROBIN))
    fifo_p = Process(target=generate_data, args=(OUTPUT_DATA_FILE_FIFO, SchedulerType.FIFO))

    rr_p.start()
    fifo_p.start()

    rr_p.join()
    fifo_p.join()


if __name__ == "__main__":
    main()