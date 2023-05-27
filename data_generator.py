from simulator import simulator, SCHEDULER_TYPE, SERVICE_TIME_DIST, SCALLING_GOVERNORS
from multiprocessing import Process
import csv

NUM_CPUS = 1
MAX_THREADS = 200
MAX_FREQ = 2000000  # 2 MHz
MIN_FREQ = 1000000  # 1 MHz
TSN_LATENCY = 100  # 100 cycles
NUM_USERS = list(range(15, 241, 15))  # 15, 30, 45, ..., 240
THINK_TIME = 10000000  # 10 seconds
AVG_INTERARRIVAL_TIME = 100000  # 0.1 seconds
AVG_SERVICE_TIME = 80000  # 80 milliseconds
TIMEOUT = 2000000  # 2 seconds
RETRY_PROB = 0.5  # 50% chance of retrying
RETRY_DELAY = 5000  # 5 milliseconds
CTX_SWITCH_OVERHEAD = 1000  # 1000 cycles
QUANTA_BURST_TIME = 2000000  # 20000 cycles
MAX_QUEUE_SIZE = 2000  # 2000 tasks
SCHEDULER = SCHEDULER_TYPE.ROUNDROBIN  # Round Robin
TIME_DIST = SERVICE_TIME_DIST.EXPONENTIAL  # Exponential distribution
SIMULATION_TIME = 300000000  # 300 seconds

OUTPUT_DATA_FILE_PERFORMANCE = "results/sim_data_performance.csv"
OUTPUT_DATA_FILE_POWERSAVE = "results/sim_data_powersave.csv"
OUTPUT_DATA_FILE_SCHEDUTIL = "results/sim_data_schedutil.csv"

# GOVERNOR = SCALLING_GOVERNORS.SCHEDUTIL  # Schedutil governor
# FREQUENCY = 0


NUM_RUNS = 10

SEEDS = [
    1,
    1973272912,
    281629770,
    20006270,
    1280689831,
    2096730329,
    1933576050,
    913566091,
    246780520,
    1363774876,
    604901985,
    1511192140,
    1259851944,
]


CSV_COLUMNS = (
    ["NUM_USERS"]
    + [f"RESPONSE_TIMES_RUN_{i+1}" for i in range(NUM_RUNS)]
    + [f"GOODPUT_RUN_{i+1}" for i in range(NUM_RUNS)]
    + [f"BADPUT_RUN_{i+1}" for i in range(NUM_RUNS)]
    + [f"DROP_RATE_RUN_{i+1}" for i in range(NUM_RUNS)]
    + [f"UTIL_RUN_{i+1}" for i in range(NUM_RUNS)]
    + [f"AVG_QUEUE_LEN_RUN_{i+1}" for i in range(NUM_RUNS)]
    + [f"QUEUE_WAIT_RUN_{i+1}" for i in range(NUM_RUNS)]
)


def print_run_config(f, governor: SCALLING_GOVERNORS):
    f.writerow(
        [
            "SCALLING_GOVERNOR",
            str(governor),
            "MAX_FREQ",
            str(MAX_FREQ),
            "MIN_FREQ",
            str(MIN_FREQ),
            "MAX_THREADS",
            str(MAX_THREADS),
            "NUM_CPUS",
            str(NUM_CPUS),
            "THINK_TIME",
            str(THINK_TIME),
            "AVG_SERVICE_TIME",
            str(AVG_SERVICE_TIME),
            "QUANTA_BURST_TIME",
            str(QUANTA_BURST_TIME),
            "AVG_INTERARRIVAL_TIME",
            str(AVG_INTERARRIVAL_TIME),
        ]
    )

    f.writerow(
        [
            "TIMEOUT",
            str(TIMEOUT),
            "SIMULATION_TIME",
            str(SIMULATION_TIME),
            "CTX_SWITCH_OVERHEAD",
            str(CTX_SWITCH_OVERHEAD),
            "MAX_QUEUE_SIZE",
            str(MAX_QUEUE_SIZE),
            "RETRY_PROB",
            str(RETRY_PROB),
            "RETRY_DELAY",
            str(RETRY_DELAY),
            "SCHEDULER",
            SCHEDULER.name,
            "SERVICE_TIME_DIST",
            TIME_DIST.name,
        ]
    )


def write_data_row(
    f,
    n_users,
    response_times,
    good_puts,
    bad_puts,
    request_drop_rates,
    core_utilsations,
    num_retries_per_task,
    avg_waiting_time_in_queue,
):
    f.writerow(
        [str(n_users)]
        + list(map(str, response_times))
        + list(map(str, good_puts))
        + list(map(str, bad_puts))
        + list(map(str, request_drop_rates))
        + list(map(str, core_utilsations))
        + list(map(str, num_retries_per_task))
        + list(map(str, avg_waiting_time_in_queue))
    )


def generate_data(output_file_name: str, governor: SCALLING_GOVERNORS):
    file = open(output_file_name, "w", newline="")
    writer = csv.writer(file)

    print_run_config(writer, governor)
    writer.writerow(CSV_COLUMNS)
    for n_user in NUM_USERS:
        response_times = []
        good_puts = []
        bad_puts = []
        request_drop_rates = []
        core_utilsations = []
        avg_queue_length = []
        avg_waiting_time_in_queue = []

        for i in range(NUM_RUNS):
            sim = simulator(NUM_CPUS, MAX_THREADS, MAX_FREQ, MIN_FREQ, TSN_LATENCY)
            sim.run_simulation(
                n_user,
                THINK_TIME,
                AVG_INTERARRIVAL_TIME,
                AVG_SERVICE_TIME,
                TIMEOUT,
                RETRY_PROB,
                RETRY_DELAY,
                CTX_SWITCH_OVERHEAD,
                QUANTA_BURST_TIME,
                MAX_QUEUE_SIZE,
                SCHEDULER,
                TIME_DIST,
                governor,
                0,
                SIMULATION_TIME,
                SEEDS[i],
                i,
                False,
                "nothing",
            )

            response_times.append(sim.get_avg_response_time() / 1000000)
            good_puts.append(sim.get_good_put() * 1000000)
            bad_puts.append(sim.get_bad_put() * 1000000)
            request_drop_rates.append(sim.get_dropped_requests() * 1000000)
            core_utilsations.append(sim.get_core_utilisation())
            avg_queue_length.append(sim.get_average_queue_length())
            avg_waiting_time_in_queue.append(
                sim.get_average_waiting_time_in_queue() / 1000000
            )

        write_data_row(
            writer,
            n_user,
            response_times,
            good_puts,
            bad_puts,
            request_drop_rates,
            core_utilsations,
            avg_queue_length,
            avg_waiting_time_in_queue,
        )
    file.close()


def main():
    perf_p = Process(
        target=generate_data,
        args=(OUTPUT_DATA_FILE_PERFORMANCE, SCALLING_GOVERNORS.PERFORMANCE),
    )
    power_p = Process(
        target=generate_data,
        args=(OUTPUT_DATA_FILE_POWERSAVE, SCALLING_GOVERNORS.POWERSAVE),
    )
    sched_p = Process(
        target=generate_data,
        args=(OUTPUT_DATA_FILE_SCHEDUTIL, SCALLING_GOVERNORS.SCHEDUTIL),
    )

    perf_p.start()
    power_p.start()
    sched_p.start()

    perf_p.join()
    power_p.join()
    sched_p.join()


if __name__ == "__main__":
    main()
