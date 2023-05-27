from dataclasses import dataclass, field
from enum import Enum, IntEnum
from math import log
from lcgrand import *
from abc import ABC, abstractmethod
import heapq
import csv


def expon(mean: int, generator: lcg) -> int:
    return round((-mean) * log(generator.lcgrand()))


def uniform(mean: int, genetator: lcg) -> int:
    return round(2 * mean * genetator.lcgrand())


class THREAD_STATE(Enum):
    CREATED = 1
    READY = 2
    RUNNING = 3
    DONE = 4


class CPU_STATE(Enum):
    IDLE = 0
    BUSY = 1


class USER_STATE(Enum):
    READY = 0
    WAITING = 1


class TASK_COMPLETION_STATE(Enum):
    SUCCESS = 0
    TIMEOUT = 1
    DROPPED = 2


class EVENT_TYPE(IntEnum):
    DEQUEUE = 0
    ARRIVAL = 1
    EXECUTION = 2
    CTXSWITCH = 3


class DIST_TYPE(Enum):
    CONSTANT = 0
    UNIFORM = 1
    EXPONENTIAL = 2


class SCALLING_GOVERNORS(Enum):
    PERFORMANCE = 0
    POWERSAVE = 1
    USERSPACE = 2
    SCHEDUTIL = 3


@dataclass
class counters:
    # Counter for thread ID
    TASKIDCOUNTER: int = field(default=0, init=False)
    THREADIDCOUNTER: int = field(default=0, init=False)

    def get_task_id_counter(self) -> int:
        """Returns and increments task id counter"""
        tid = self.TASKIDCOUNTER
        self.TASKIDCOUNTER += 1
        return tid

    def get_thread_id_counter(self) -> int:
        """Returns and increments thread id counter"""
        tid = self.THREADIDCOUNTER
        self.THREADIDCOUNTER += 1
        return tid


# The task class represents an individual task
@dataclass
class task:
    # Attributes of the task class
    task_id: int
    user_id: int
    arrival_time: int
    burst_cycles: int
    timeout_duration: int
    remaining_cycles: int
    num_retries: int = field(default=0, init=False)
    completion_state: TASK_COMPLETION_STATE = field(init=False)
    departure_time: int = field(init=False)
    queue_length_observed: int = field(init=False, default=0)
    waiting_time: int = field(init=False, default=0)
    burst_time: int = field(init=False, default=0)
    num_context_switches: int = field(init=False, default=0)
    total_context_switch_overhead: int = field(init=False, default=0)
    thread_id: int = field(init=False)
    cpu_id: int = field(init=False)
    waiting_time_for_thread: int = field(default=0, init=False)


# The CPU class represents a core
@dataclass
class cpu:
    cpu_id: int
    cpu_max_freq: int
    cpu_min_freq: int
    cpu_transition_latency: int
    cpu_cur_freq: int = field(init=False)
    cpu_state: CPU_STATE = field(default=CPU_STATE.IDLE, init=False)
    cpu_cur_time: int = field(default=0, init=False)
    total_execution_time: int = field(default=0, init=False)


# The user class
@dataclass
class user:
    user_id: int
    think_time: int
    avg_interarrival_time: int
    avg_service_time: int
    avg_cpu_freq: int
    timeout_duration: int
    retry_prob: float
    retry_time: int
    count: counters
    rand_gen: lcg
    service_time_dist: DIST_TYPE
    associated_task: task = field(init=False)
    user_state: USER_STATE = field(default=USER_STATE.READY, init=False)
    completed_tasks: list[task] = field(init=False, default_factory=list)

    def __post_init__(self):
        """Add initial task for the user"""
        self.create_task(expon(self.avg_interarrival_time, self.rand_gen))

    def create_task(self, arrival_time: int) -> None:
        """Creates a new task for the given arrival time"""
        # get service time from distribution
        service_time = (
            self.avg_service_time
            if self.service_time_dist == DIST_TYPE.CONSTANT
            else uniform(self.avg_service_time, self.rand_gen)
            if self.service_time_dist == DIST_TYPE.UNIFORM
            else expon(self.avg_service_time, self.rand_gen)
        )

        # 50% of the timeout is minimum and rest is exponential
        min_timeout = self.timeout_duration // 2
        var_timeout = expon(self.timeout_duration - min_timeout, self.rand_gen)
        total_timeout = min_timeout + var_timeout

        self.associated_task = task(
            self.count.get_task_id_counter(),
            self.user_id,
            arrival_time,
            int(service_time * self.avg_cpu_freq / 1000000),  # burst cycles
            total_timeout,
            int(service_time * self.avg_cpu_freq / 1000000),  # remaining cycles
        )

    def send_request(self) -> task:
        """Sends the task to server"""
        # set the user state to waiting
        self.user_state = USER_STATE.WAITING

        # print trace
        if __debug__:
            print(
                f"|{'ARRIVAL':15s}|{self.associated_task.arrival_time:<10d}|{f'USERID {self.user_id}':10s}|{f'TASKID {self.associated_task.task_id}':10s}|{f'BURST {self.associated_task.burst_time}':10s}|"
            )
        return self.associated_task

    def dropped_request(self) -> None:
        """Method to handle request drops"""
        # with probability p retry the request
        if self.rand_gen.lcgrand() <= self.retry_prob:
            # retry
            self.associated_task.arrival_time += expon(self.retry_time, self.rand_gen)
            self.associated_task.num_retries += 1
            # change the state of the user
            self.user_state = USER_STATE.READY
            if __debug__:
                print(
                    f"|{'RETRY':15s}|{self.associated_task.arrival_time:<10d}|{f'USERID {self.user_id}':10s}|{f'TASKID {self.associated_task.task_id}':10s}|{f'RETRYN {self.associated_task.num_retries}':10s}|"
                )
        else:
            if __debug__:
                print(
                    f"|{'DROPPED':15s}|{self.associated_task.arrival_time:<10d}|{f'USERID {self.user_id}':10s}|{f'TASKID {self.associated_task.task_id}':10s}|{f'RETRYN {self.associated_task.num_retries}':10s}|"
                )

            # mark the request as dropped
            self.associated_task.completion_state = TASK_COMPLETION_STATE.DROPPED
            self.associated_task.departure_time = self.associated_task.arrival_time
            self.completed_tasks.append(self.associated_task)

            # generate a new request
            self.create_task(self.completed_tasks[-1].departure_time + self.think_time)
            self.user_state = USER_STATE.READY

    def request_completed(self) -> None:
        if __debug__:
            print(
                f"|{self.associated_task.completion_state.name:15s}|{self.associated_task.departure_time:<10d}|{f'USERID {self.user_id}':10s}|{f'TASKID {self.associated_task.task_id}':10s}|{f'BURST {self.associated_task.burst_time}':10s}|"
            )

        # add current task to completed tasks list
        self.completed_tasks.append(self.associated_task)

        # think time is combination of min fixed + variable time
        think_time = round(0.8 * self.think_time) + expon(
            0.2 * self.think_time, self.rand_gen
        )

        # create a new task
        self.create_task(self.completed_tasks[-1].departure_time + think_time)

        # set user state to ready
        self.user_state = USER_STATE.READY


# Class to manage list of users
@dataclass
class user_list:
    users: dict[int, user] = field(default_factory=dict)

    def add_user(self, user: user) -> None:
        """Adds a new user to the list"""
        self.users[user.user_id] = user

    def get_all_ready_users(self) -> list[user]:
        """Returns the list of all ready users"""
        return [x for x in self.users.values() if x.user_state == USER_STATE.READY]

    def get_user(self, user_id: int) -> user:
        """Returns the user with the given user id"""
        return self.users[user_id]

    def get_all_users(self) -> list[user]:
        """Returns the list of all users"""
        return self.users.values()


# The thread class represent a thread.
# Each thread is assigned a task
@dataclass
class thread:
    thread_id: int
    cpu_id: int
    associated_task: task
    thread_state: THREAD_STATE = field(default=THREAD_STATE.CREATED, init=False)

    def thread_completed(
        self,
        current_cpu_time: int,
        users: user_list,
    ) -> int:
        """Thread has completed execution
        Returns the associated user id"""
        # Mark thread state as done
        self.thread_state = THREAD_STATE.DONE
        # update task departure time
        self.associated_task.departure_time = current_cpu_time
        # update total waiting time
        self.associated_task.waiting_time = (
            self.associated_task.departure_time
            - self.associated_task.arrival_time
            - self.associated_task.burst_time
        )
        # update the turn aroung time
        tat = self.associated_task.departure_time - self.associated_task.arrival_time

        self.associated_task.completion_state = (
            TASK_COMPLETION_STATE.SUCCESS
            if self.associated_task.timeout_duration >= tat
            else TASK_COMPLETION_STATE.TIMEOUT
        )

        # update the user
        users.get_user(self.associated_task.user_id).request_completed()
        return self.associated_task.user_id


@dataclass
class scheduler(ABC):
    context_switch_overhead: int
    associated_cpu: cpu
    max_thread_queue_size: int
    users_list: user_list
    counters: counters
    scalling_governor: SCALLING_GOVERNORS
    exec_thread_queue: list[thread] = field(default_factory=list, init=False)
    cur_time_quanta: int = field(default=0, init=False)
    executing_thread_idx: int = field(default=-1, init=False)
    last_execution_time: int = field(default=0, init=False)
    last_context_switch_time: int = field(default=0, init=False)

    def is_thread_queue_full(self) -> bool:
        """Returns true if thread queue is full"""
        return len(self.exec_thread_queue) == self.max_thread_queue_size

    def get_next_context_switch_time(self) -> int:
        """Returns the context switch time of next task"""
        if self.associated_cpu.cpu_state == CPU_STATE.IDLE:
            return -1
        return self.associated_cpu.cpu_cur_time + int(
            self.cur_time_quanta / self.associated_cpu.cpu_cur_freq * 1000000
        )

    def execute_current_thread(self):
        """Executes the current thread"""
        if self.executing_thread_idx != -1:
            # set thread state to running
            self.exec_thread_queue[
                self.executing_thread_idx
            ].thread_state = THREAD_STATE.RUNNING
            # update cpu stats
            if __debug__:
                print(
                    f'|{"EXECUTE":15s}|{self.associated_cpu.cpu_cur_time:<10d}|{f"CPUID {self.associated_cpu.cpu_id}":10s}|{f"TID {self.exec_thread_queue[self.executing_thread_idx].thread_id}":10s}|{f"FOR {self.cur_time_quanta}":10s}|'
                )
            self.associated_cpu.cpu_cur_time += int(
                self.cur_time_quanta / self.associated_cpu.cpu_cur_freq * 1000000
            )
            self.associated_cpu.total_execution_time += (
                self.cur_time_quanta / self.associated_cpu.cpu_cur_freq * 1000000
            )
            # update task remaining time and add burst time
            self.exec_thread_queue[
                self.executing_thread_idx
            ].associated_task.remaining_cycles -= self.cur_time_quanta
            self.exec_thread_queue[
                self.executing_thread_idx
            ].associated_task.burst_time += (
                self.cur_time_quanta / self.associated_cpu.cpu_cur_freq * 1000000
            )

    def context_switch(self) -> user:
        """Gets the next task to schedule
        Returns user if its task is completed"""
        next_idx = -1
        completed_user = None
        if __debug__:
            print(
                f'|{"CTXSWITCH":15s}|{self.associated_cpu.cpu_cur_time:<10d}|{f"CPUID {self.associated_cpu.cpu_id}":10s}|{f"TID {self.exec_thread_queue[self.executing_thread_idx].thread_id}":10s}|{f"TASKID {self.exec_thread_queue[self.executing_thread_idx].associated_task.task_id}":10s}|'
            )

        # if the current thread has completed its execution
        # then mark it as completed and remove from queue
        if (
            self.exec_thread_queue[
                self.executing_thread_idx
            ].associated_task.remaining_cycles
            <= 0
        ):
            # call compeletion handler
            uid = self.exec_thread_queue[self.executing_thread_idx].thread_completed(
                self.associated_cpu.cpu_cur_time, self.users_list
            )
            completed_user = self.users_list.get_user(uid)

            # remove thread from queue
            del self.exec_thread_queue[self.executing_thread_idx]

            # since item is deleted, the current index points
            # to the next Thread
            next_idx = (
                self.executing_thread_idx % len(self.exec_thread_queue)
                if len(self.exec_thread_queue) != 0
                else -1
            )

        elif len(self.exec_thread_queue) == 1:
            # if there is only one task then execute the same task without context switch overhead
            next_idx = self.executing_thread_idx
        else:
            # this task is context switching out increases number of context switches
            self.exec_thread_queue[
                self.executing_thread_idx
            ].associated_task.num_context_switches += 1

            # update context switch overhead
            self.exec_thread_queue[
                self.executing_thread_idx
            ].associated_task.total_context_switch_overhead += (
                self.context_switch_overhead
                / self.associated_cpu.cpu_cur_freq
                * 1000000
            )

            # set thread state to ready
            self.exec_thread_queue[
                self.executing_thread_idx
            ].thread_state = THREAD_STATE.READY

            # increment index
            next_idx = (self.executing_thread_idx + 1) % len(self.exec_thread_queue)

        # if thread queue is empty then return -1
        if len(self.exec_thread_queue) == 0:
            # set cpu to idle
            self.associated_cpu.cpu_state = CPU_STATE.IDLE
            self.cur_time_quanta = 0
            self.executing_thread_idx = -1
            return completed_user

        # add context switch overhead to the cpu time
        # if there are more than 1 threads in the queue
        if len(self.exec_thread_queue) > 1:
            self.associated_cpu.cpu_cur_time += int(
                self.context_switch_overhead
                / self.associated_cpu.cpu_cur_freq
                * 1000000
            )
            # add to the cpu execution time
            self.associated_cpu.total_execution_time += (
                self.context_switch_overhead
                / self.associated_cpu.cpu_cur_freq
                * 1000000
            )

        # set the next time_quanta
        self.cur_time_quanta = self.get_next_time_quanta(
            self.exec_thread_queue[next_idx].associated_task
        )

        # set the index of next scheduled thread
        self.executing_thread_idx = next_idx

        # return completed user
        return completed_user

    @abstractmethod
    def get_next_time_quanta(self, t_task: task) -> int:
        """Returns time quanta"""
        pass

    def add_task_and_create_thread(self, t_task: task) -> bool:
        """Adds a task and creates thread
        returns true if cpu was idle before"""
        # create a thread and assign it to the cpu
        t_thread = thread(
            self.counters.get_thread_id_counter(), self.associated_cpu.cpu_id, t_task
        )
        t_task.thread_id = t_thread.thread_id
        t_task.cpu_id = self.associated_cpu.cpu_id

        # add it to the scheduler
        cpu_idle = self.add_thread(t_thread)

        if __debug__:
            print(
                f'|{"THREADCREATE":15s}|{self.associated_cpu.cpu_cur_time:<10d}|{f"CPUID {self.associated_cpu.cpu_id}":10s}|{f"TID {t_thread.thread_id}":10s}|{f"TASKID {t_task.task_id}":10s}|'
            )

        return cpu_idle

    def add_thread(self, t_thread: thread) -> bool:
        """Adds a thread to the thread queue
        returns true if cpu was idle before"""
        assert len(self.exec_thread_queue) < self.max_thread_queue_size

        # add thread in the last
        self.exec_thread_queue.append(t_thread)

        cpu_idle = False

        # if this is the first thread
        if len(self.exec_thread_queue) == 1:
            assert self.associated_cpu.cpu_state == CPU_STATE.IDLE
            cpu_idle = True
            # update current time quanta
            self.cur_time_quanta = self.get_next_time_quanta(t_thread.associated_task)
            self.executing_thread_idx = 0
            # set cpu to busy
            self.associated_cpu.cpu_state = CPU_STATE.BUSY
            # set the cpu time to the arrival time
            self.associated_cpu.cpu_cur_time = max(
                t_thread.associated_task.arrival_time, self.associated_cpu.cpu_cur_time
            )

        # update waiting in thread time
        t_thread.associated_task.waiting_time_for_thread = (
            self.associated_cpu.cpu_cur_time - t_thread.associated_task.arrival_time
        )

        # update thread state to ready
        thread.thread_state = THREAD_STATE.READY
        return cpu_idle


@dataclass
class round_robin_scheduler(scheduler):
    max_time_quanta: int

    def get_next_time_quanta(self, t_task: task) -> int:
        return min(t_task.remaining_cycles, self.max_time_quanta)


@dataclass
class fifo_scheduler(scheduler):
    def get_next_time_quanta(self, t_task: task) -> int:
        """returns time quanta"""
        return t_task.remaining_cycles


# class to represent a list of schedulers
@dataclass
class scheduler_list:
    schedulers: list[scheduler] = field(init=False, default_factory=list)
    counts_task_added: dict[int, int] = field(init=False, default_factory=dict)

    def add(self, sch: scheduler) -> None:
        """Adds a scheduler to the list"""
        self.schedulers.append(sch)
        self.counts_task_added[sch.associated_cpu.cpu_id] = 0

    def get_a_scheduler(self) -> scheduler:
        """Returns a scheduler with cpu that has fewest threads
        returns None if all scheduler is fully occupied"""

        min_q_sched = min(self.schedulers, key=lambda x: len(x.exec_thread_queue))

        if min_q_sched.is_thread_queue_full():
            return None

        # if more than 1 schedulers have min thread queue count,
        # then return the one which has least counts_task_added value

        t_scheds = [
            x
            for x in self.schedulers
            if len(x.exec_thread_queue) == len(min_q_sched.exec_thread_queue)
        ]

        target_sched = min_q_sched

        if len(t_scheds) > 1:
            target_sched = min(
                t_scheds, key=lambda x: self.counts_task_added[x.associated_cpu.cpu_id]
            )

        # increase count
        self.counts_task_added[target_sched.associated_cpu.cpu_id] += 1
        return target_sched


@dataclass
class task_queue:
    max_queue_length: int
    queue: list[task] = field(init=False, default_factory=list)

    def is_full(self) -> bool:
        """Returns if the task queue is full"""
        return self.length() == self.max_queue_length

    def length(self) -> int:
        """Returns the length of the task queue"""
        return len(self.queue)

    def enqueue(self, t_task: task) -> None:
        """Adds a task to the queue"""
        assert len(self.queue) < self.max_queue_length
        self.queue.insert(0, t_task)

    def peek(self) -> task:
        """Return the top task without popping"""
        return self.queue[-1]

    def dequeue(self) -> task:
        """Pops and returns task in FIFO order"""
        assert len(self.queue) != 0
        return self.queue.pop()


@dataclass(order=True)
class event:
    # class to represent event details
    event_time: int
    event_type: EVENT_TYPE
    associated_object: object = field(compare=False)

    def __eq__(self, __o: object) -> bool:
        """all fields are equal"""
        return (
            self.event_time == __o.event_time
            and self.event_type == __o.event_type
            and id(self.associated_object) == id(__o.associated_object)
        )


@dataclass
class event_list:
    events: list[event] = field(init=False, default_factory=list)

    def add_event(self, event: event) -> None:
        """Adds a new event to the pq"""
        ## if same event is not already in the heap then add
        if event not in self.events:
            heapq.heappush(self.events, event)

    def get_next_event(self) -> event:
        """returns the top event"""
        return heapq.heappop(self.events)


class event_handlers:
    @staticmethod
    def arrival_event_handler(
        event_time: int,
        t_user: user,
        schedulers: scheduler_list,
        queue_of_task: task_queue,
        events: event_list,
    ) -> None:
        # apply arrival logic
        t_task = t_user.send_request()

        # check if queue is full
        if queue_of_task.is_full():
            # then drop the request
            t_user.dropped_request()
        else:
            if __debug__:
                print(
                    f"|{'ENQUEUE':15s}|{event_time:<10}|{f'USERID {t_user.user_id}':10s}|{f'TASKID {t_task.task_id}':10s}|{f'QLEN {queue_of_task.length()}':10s}"
                )
            # record the queue length
            t_task.queue_length_observed = queue_of_task.length()
            # add task to the queue
            queue_of_task.enqueue(t_user.associated_task)
            # check if any cpu is free
            # get a free cpu
            free_sched = schedulers.get_a_scheduler()
            # TODO: check if this is working correctly
            if free_sched != None:
                # create a deque event at the next context switch time
                events.add_event(
                    event(
                        max(event_time, free_sched.get_next_context_switch_time()),
                        EVENT_TYPE.DEQUEUE,
                        free_sched,
                    )
                )

    @staticmethod
    def execution_event_handler(
        event_time: int, sched: scheduler, events: event_list
    ) -> None:
        # apply execution logic
        sched.execute_current_thread()
        # add context switch event to the events list
        events.add_event(
            event(sched.associated_cpu.cpu_cur_time, EVENT_TYPE.CTXSWITCH, sched)
        )

    @staticmethod
    def context_swtich_event_handler(
        event_time: int,
        sched: scheduler,
        queue_of_task: task_queue,
        schedulers: scheduler_list,
        list_of_users: user_list,
        events: event_list,
    ) -> None:
        if (sched.associated_cpu.cpu_cur_time - sched.last_context_switch_time) > (
            1000
            * sched.associated_cpu.cpu_transition_latency
            / sched.associated_cpu.cpu_cur_freq
            * 1000000
        ):
            utilization = (
                sched.associated_cpu.total_execution_time - sched.last_execution_time
            ) / (sched.associated_cpu.cpu_cur_time - sched.last_context_switch_time)

            if sched.scalling_governor == SCALLING_GOVERNORS.SCHEDUTIL:
                # update the cpu frequency
                sched.associated_cpu.cpu_cur_freq = max(
                    sched.associated_cpu.cpu_min_freq,
                    min(
                        sched.associated_cpu.cpu_max_freq,
                        (1.25 * utilization * sched.associated_cpu.cpu_max_freq),
                    ),
                )
                sched.associated_cpu.cpu_cur_time += int(
                    sched.associated_cpu.cpu_transition_latency
                    / sched.associated_cpu.cpu_cur_freq
                    * 1000000
                )

                sched.last_execution_time = sched.associated_cpu.total_execution_time
                sched.last_context_switch_time = sched.associated_cpu.cpu_cur_time

        completed_user = sched.context_switch()

        # add the next execution event
        if sched.associated_cpu.cpu_state != CPU_STATE.IDLE:
            events.add_event(
                event(sched.associated_cpu.cpu_cur_time, EVENT_TYPE.EXECUTION, sched)
            )

        # check if task queue is not empty
        if queue_of_task.length() != 0 and not sched.is_thread_queue_full():
            # check if cpu is free
            # add dequeue event to the list
            events.add_event(
                event(
                    max(
                        queue_of_task.peek().arrival_time,
                        sched.associated_cpu.cpu_cur_time,
                    ),
                    EVENT_TYPE.DEQUEUE,
                    sched,
                )
            )

        # check if user is completed, then add next arrival event
        if completed_user != None:
            events.add_event(
                event(
                    completed_user.associated_task.arrival_time,
                    EVENT_TYPE.ARRIVAL,
                    completed_user,
                )
            )

    @staticmethod
    def deque_event_handler(
        event_time: int, sched: scheduler, queue_of_task: task_queue, events: event_list
    ) -> None:
        # if scheduler is still not free, then don't deque
        if sched.is_thread_queue_full() or queue_of_task.length() == 0:
            return
        # deque from the task queue and create a thread in scheduler
        if __debug__:
            print(
                f"|{'DEQUEUE':15s}|{event_time:<10d}|{f'CPUID {sched.associated_cpu.cpu_id}':10s}|{f'TASKID {queue_of_task.peek().task_id}':10s}|{f'QLEN {queue_of_task.length()-1}':10s}"
            )
        ts = queue_of_task.dequeue()

        cpu_idle = sched.add_task_and_create_thread(ts)
        if cpu_idle:
            events.add_event(
                event(sched.associated_cpu.cpu_cur_time, EVENT_TYPE.EXECUTION, sched)
            )
