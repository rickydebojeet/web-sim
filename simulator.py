from classes import *


class SCHEDULER_TYPE(Enum):
    ROUNDROBIN = 0
    FIFO = 1


class SERVICE_TIME_DIST(Enum):
    CONSTANT = 0
    UNIFORM = 1
    EXPONENTIAL = 2


class simulator:
    def __init__(
        self,
        num_cpus: int,
        max_threads: int,
        max_freq: int,
        min_freq: int,
        transition_latency: int,
    ) -> None:
        """Initializes the simulator with the given parameters"""
        self.num_cpus = num_cpus
        self.max_threads = max_threads
        self.cpus = [
            cpu(cpu_id, max_freq, min_freq, transition_latency)
            for cpu_id in range(num_cpus)
        ]
        self.counters = counters()
        self.user_list = user_list()
        self.avg_cpu_freq = (max_freq + min_freq) / 2

    def run_simulation(
        self,
        num_users,
        think_time,
        avg_interarrival_time,
        avg_service_time,
        timeout,
        retry_prob,
        retry_time,
        ctx_switch_overhead,
        ctx_switch_time,
        max_queue_size,
        scheduler_type: SCHEDULER_TYPE,
        service_time_dist: SERVICE_TIME_DIST,
        scaling_governor: SCALLING_GOVERNORS,
        frequency,
        simulation_time,
        rseed=1,
        rstream=0,
        generate_freq_csv=False,
        file_name="freq_file.csv",
    ) -> None:
        """Runs the simulation"""

        if generate_freq_csv:
            self.file = open(file_name, "w", newline="")
            self.writer = csv.writer(self.file)
            self.writer.writerow(
                ["Timestamp"] + [f"CPU_{i}_FREQ" for i in range(len(self.cpus))]
            )

        self.simulation_time = simulation_time

        # Initialize the frequency of the cpus
        if scaling_governor == SCALLING_GOVERNORS.PERFORMANCE:
            for i in self.cpus:
                i.cpu_cur_freq = i.cpu_max_freq
        elif (
            scaling_governor == SCALLING_GOVERNORS.POWERSAVE
            or scaling_governor == SCALLING_GOVERNORS.SCHEDUTIL
        ):
            for i in self.cpus:
                i.cpu_cur_freq = i.cpu_min_freq
        elif scaling_governor == SCALLING_GOVERNORS.USERSPACE and frequency != 0:
            for i in self.cpus:
                if frequency >= i.cpu_min_freq and frequency <= i.cpu_max_freq:
                    i.cpu_cur_freq = frequency
                else:
                    assert False, "Invalid frequency"
        else:
            assert False, "Invalid scaling governor"

        # Initialize the random number generator
        rand_gen = lcg()
        rand_gen.lcgrandst(rseed, rstream)
        rand_gen.lcgSetActiveStream(rstream)

        # Initialize the user
        for i in range(num_users):
            time_dist = DIST_TYPE(service_time_dist.value)
            t_user = user(
                i,
                think_time,
                avg_interarrival_time,
                avg_service_time,
                self.avg_cpu_freq,
                timeout,
                retry_prob,
                retry_time,
                self.counters,
                rand_gen,
                time_dist,
            )
            self.user_list.add_user(t_user)

        events = event_list()

        # initally add all arrival events to the event list
        for i in self.user_list.get_all_ready_users():
            events.add_event(
                event(i.associated_task.arrival_time, EVENT_TYPE.ARRIVAL, i)
            )

        schedulers = scheduler_list()

        # create scheduler for each cpu
        if scheduler_type == SCHEDULER_TYPE.ROUNDROBIN:
            for i in self.cpus:
                schedulers.add(
                    round_robin_scheduler(
                        ctx_switch_overhead,
                        i,
                        self.max_threads,
                        self.user_list,
                        self.counters,
                        scaling_governor,
                        ctx_switch_time,
                    )
                )
        elif scheduler_type == SCHEDULER_TYPE.FIFO:
            for i in self.cpus:
                schedulers.add(
                    fifo_scheduler(
                        ctx_switch_overhead,
                        i,
                        self.max_threads,
                        self.user_list,
                        self.counters,
                        scaling_governor,
                    )
                )
        else:
            assert False, "Invalid scheduler type"

        queue_of_tasks = task_queue(max_queue_size)

        while True:
            # get the next event
            next_event = events.get_next_event()

            # check if the simulation time has elapsed
            if next_event.event_time > simulation_time:
                break

            if generate_freq_csv:
                self.writer.writerow(
                    [next_event.event_time] + [i.cpu_cur_freq for i in self.cpus]
                )

            # check if the event is an arrival event
            if next_event.event_type == EVENT_TYPE.ARRIVAL:
                event_handlers.arrival_event_handler(
                    next_event.event_time,
                    next_event.associated_object,
                    schedulers,
                    queue_of_tasks,
                    events,
                )

            # check if the event is a Execution event
            elif next_event.event_type == EVENT_TYPE.EXECUTION:
                event_handlers.execution_event_handler(
                    next_event.event_time, next_event.associated_object, events
                )

            # check if the event is context switch event
            elif next_event.event_type == EVENT_TYPE.CTXSWITCH:
                event_handlers.context_swtich_event_handler(
                    next_event.event_time,
                    next_event.associated_object,
                    queue_of_tasks,
                    schedulers,
                    self.user_list,
                    events,
                )

            # check if the event is a dequeue event
            elif next_event.event_type == EVENT_TYPE.DEQUEUE:
                event_handlers.deque_event_handler(
                    next_event.event_time,
                    next_event.associated_object,
                    queue_of_tasks,
                    events,
                )
        if generate_freq_csv:
            self.file.close()

    def get_avg_response_time(self):
        """Returns the average response time of all the users"""
        total_requests = 0
        total_time = 0

        for i in self.user_list.get_all_users():
            for com_task in i.completed_tasks:
                if com_task.completion_state != TASK_COMPLETION_STATE.DROPPED:
                    total_requests += 1
                    total_time += com_task.departure_time - com_task.arrival_time

        # return the average response time
        return total_time / total_requests

    def get_good_put(self) -> float:
        """Get the goodput after simulation"""
        completed_requests = 0
        for i in self.user_list.get_all_users():
            for com_task in i.completed_tasks:
                if com_task.completion_state == TASK_COMPLETION_STATE.SUCCESS:
                    completed_requests += 1

        # divide by simulation time
        return completed_requests / self.simulation_time

    def get_bad_put(self) -> float:
        """Get the badput after simulation"""
        timed_out_requests = 0
        for i in self.user_list.get_all_users():
            for com_task in i.completed_tasks:
                if com_task.completion_state == TASK_COMPLETION_STATE.TIMEOUT:
                    timed_out_requests += 1

        # divide by simulation time
        return timed_out_requests / self.simulation_time

    def get_dropped_requests(self) -> float:
        """Get dropped request rate after simulation"""
        dropped_requests = 0
        for i in self.user_list.get_all_users():
            for com_task in i.completed_tasks:
                if com_task.completion_state == TASK_COMPLETION_STATE.DROPPED:
                    dropped_requests += 1

        # divide by simulation time
        return dropped_requests / self.simulation_time

    def get_core_utilisation(self) -> float:
        """Get the core utilisation after simulation"""
        return (
            sum([cpu.total_execution_time / cpu.cpu_cur_time for cpu in self.cpus])
            / self.num_cpus
        )

    def get_average_queue_length(self) -> float:
        """Get the average queue length after simulation"""
        total_length = 0
        total_tasks = 0
        for i in self.user_list.get_all_users():
            for com_task in i.completed_tasks:
                total_length += com_task.queue_length_observed
                total_tasks += 1
        return total_length / total_tasks

    def get_average_waiting_time_in_queue(self) -> float:
        """Get the average waiting time in queue after simulation"""
        total_time = 0
        total_requests = 0
        for i in self.user_list.get_all_users():
            for com_task in i.completed_tasks:
                if com_task.completion_state != TASK_COMPLETION_STATE.DROPPED:
                    total_requests += 1
                    total_time += com_task.waiting_time_for_thread
        return total_time / total_requests


def test():
    # 2 cpus, 100 threads, 6Mhz max freq, 1Mhz min freq, 1000 cycles (1ms) transition latency
    sim = simulator(2, 100, 6000000, 1000000, 1000)
    # 250 users, 10s think time, 0.1s avg interarrival time,
    # 80ms avg service time, 2s timeout, 0.5 retry probability,
    # 5ms retry time, 1000 cycles ctx switch overhead, 20000 cycles ctx switch time,
    # 200 max queue size, round robin scheduler, exponential service time dist,
    # performance scaling governor, - frequency, 300s simulation time
    sim.run_simulation(
        500,
        10000000,
        100000,
        80000,
        2000000,
        0.5,
        5000,
        1000,
        20000,
        200,
        SCHEDULER_TYPE.ROUNDROBIN,
        SERVICE_TIME_DIST.EXPONENTIAL,
        SCALLING_GOVERNORS.SCHEDUTIL,
        0,
        300000000,
        1973272912,
        1,
        True,
        "2corehighload.csv",
    )

    print(f"Response time: {sim.get_avg_response_time()/1000000} s")
    goodput = sim.get_good_put()
    badput = sim.get_bad_put()
    throughput = goodput + badput
    print(f"Goodput: {goodput*1000000} req/s")
    print(f"Badput: {badput*1000000} req/s")
    print(f"Throughput: {throughput*1000000} req/s")
    print(f"Request Drop Rate: {sim.get_dropped_requests()*1000000} req/s")
    print(f"Utilisation: {sim.get_core_utilisation()}")
    print(f"Average Queue Length: {sim.get_average_queue_length()}")
    print(
        f"Avg waiting time in queue: {sim.get_average_waiting_time_in_queue()/1000000} s"
    )


if __name__ == "__main__":
    test()
