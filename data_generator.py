from simulator import simulator, SchedulerType
import csv

NUM_USERS = [ 1, 2, 3, 4, 5, 10 ] + list(range(15, 315, 15))
CLOCKS_PER_SEC = 1000000
MAX_THREADS = 50
NUM_CPUS = 2
THINK_TIME = 10000000
AVG_SERVICE_TIME = 80000
QUANTA_BURST_TIME = 20000
AVG_INTERARRIVAL_TIME = 100000
TIMEOUT = 2000000
SIMULATION_TIME = 300000000
CTX_SWITCH_OVERHEAD = 1000
MAX_QUEUE_SIZE = 200
RETRY_PROB = 0.5
RETRY_DELAY = 50000

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

def main():

    rr_file = open(OUTPUT_DATA_FILE_RR, "w", newline='')
    fifo_file = open(OUTPUT_DATA_FILE_FIFO, "w", newline='')

    rr_writer = csv.writer(rr_file)
    fifo_writer = csv.writer(fifo_file)

    print_run_config(rr_writer, SchedulerType.ROUNDROBIN)
    print_run_config(fifo_writer, SchedulerType.FIFO)

    rr_writer.writerow(CSV_COLUMNS)
    fifo_writer.writerow(CSV_COLUMNS)


    for n_user in NUM_USERS:        
        responseTimesRR = []
        goodPutsRR = []
        badPutsRR = []
        requestDropRatesRR = []
        coreUtilsationsRR = []
        numRetriesPerTaskRR = []
        avgWaitingTimeInQueueRR = []

        responseTimesFIFO = []
        goodPutsFIFO = []
        badPutsFIFO = []
        requestDropRatesFIFO = []
        coreUtilsationsFIFO = []
        numRetriesPerTaskFIFO = []
        avgWaitingTimeInQueueFIFO = []

        for i in range(NUM_RUNS):
            simRR = simulator(NUM_CPUS, MAX_THREADS)

            simRR.run_simulation(n_user, THINK_TIME, AVG_SERVICE_TIME, QUANTA_BURST_TIME, AVG_INTERARRIVAL_TIME, TIMEOUT, SIMULATION_TIME, CTX_SWITCH_OVERHEAD, MAX_QUEUE_SIZE, RETRY_PROB, RETRY_DELAY, SchedulerType.ROUNDROBIN, SEEDS[i], i)

            responseTimesRR.append(simRR.getAvgResponseTime() / CLOCKS_PER_SEC)
            goodPutsRR.append(simRR.getGoodPut() * CLOCKS_PER_SEC)
            badPutsRR.append(simRR.getBadPut() * CLOCKS_PER_SEC)
            requestDropRatesRR.append(simRR.getRequestDropRate() * CLOCKS_PER_SEC)
            coreUtilsationsRR.append(simRR.getCoreUtilisation())
            numRetriesPerTaskRR.append(simRR.getAvgQueueLength())
            avgWaitingTimeInQueueRR.append(simRR.getAvgWaitingTimeInQueue() / CLOCKS_PER_SEC)

            simFIFO = simulator(NUM_CPUS, MAX_THREADS)

            simFIFO.run_simulation(n_user, THINK_TIME, AVG_SERVICE_TIME, QUANTA_BURST_TIME, AVG_INTERARRIVAL_TIME, TIMEOUT, SIMULATION_TIME, CTX_SWITCH_OVERHEAD, MAX_QUEUE_SIZE, RETRY_PROB, RETRY_DELAY, SchedulerType.FIFO, SEEDS[i], i)

            responseTimesFIFO.append(simFIFO.getAvgResponseTime() / CLOCKS_PER_SEC)
            goodPutsFIFO.append(simFIFO.getGoodPut() * CLOCKS_PER_SEC)
            badPutsFIFO.append(simFIFO.getBadPut() * CLOCKS_PER_SEC)
            requestDropRatesFIFO.append(simFIFO.getRequestDropRate() * CLOCKS_PER_SEC)
            coreUtilsationsFIFO.append(simFIFO.getCoreUtilisation())
            numRetriesPerTaskFIFO.append(simFIFO.getAvgQueueLength())
            avgWaitingTimeInQueueFIFO.append(simFIFO.getAvgWaitingTimeInQueue() / CLOCKS_PER_SEC)
        
        write_data_row(rr_writer, n_user, responseTimesRR, goodPutsRR, badPutsRR, requestDropRatesRR, coreUtilsationsRR, numRetriesPerTaskRR, avgWaitingTimeInQueueRR)
        write_data_row(fifo_writer, n_user, responseTimesFIFO, goodPutsFIFO, badPutsFIFO, requestDropRatesFIFO, coreUtilsationsFIFO, numRetriesPerTaskFIFO, avgWaitingTimeInQueueFIFO)
    
    rr_file.close()
    fifo_file.close()


if __name__ == "__main__":
    main()