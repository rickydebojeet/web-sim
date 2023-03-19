from dataclasses import dataclass, field
from enum import Enum, IntEnum
from math import log
from lcgrand import *
from abc import ABC, abstractmethod
import heapq

def exponf(mean: float, generator: lcg) -> float:
    return (-mean) * log(generator.lcgrand())

def expon(mean: int, generator: lcg) -> int:
    return round((-mean) * log(generator.lcgrand()))

def uniform(mean: int, genetator: lcg) -> int:
    return round(2 * mean * genetator.lcgrand())

class ThreadState(Enum):
    CREATED = 1
    READY = 2
    RUNNING = 3
    DONE = 4

class CPUState(Enum):
    IDLE = 0
    BUSY = 1

class UserState(Enum):
    READY = 0
    WAITING= 1

class TaskCompletionState(Enum):
    SUCCESS = 0
    TIMEOUT = 1
    DROPPED = 2

class EventType(IntEnum):
    DEQUEUE = 0
    ARRIVAL= 1
    EXECUTION = 2
    CTXSWITCH = 3

class DistType(Enum):
    CONSTANT = 0
    UNIFORM = 1
    EXPONENTIAL = 2

@dataclass
class Counters:
    # Counter for thread ID
    TASKIDCOUNTER: int = field(default=0, init=False)
    THREADIDCOUNTER: int = field(default=0, init=False)

    def getTaskIdCounter(self) -> int:
        '''Returns and increments task id counter'''
        tid = self.TASKIDCOUNTER
        self.TASKIDCOUNTER += 1
        return tid

    def getThreadIdCounter(self) -> int:
        '''Returns and increments thread id counter'''
        tid = self.THREADIDCOUNTER
        self.THREADIDCOUNTER += 1
        return tid
    

# The task class represents an individual task
@dataclass
class Task:
    # Attributes of the task class
    taskId: int
    userId: int
    arrivalTime: int
    burstTime: int
    timeoutDuration: int
    remainingTime: int
    departureTime: int = field(init=False)
    waitingTime: int = field(default=0, init=False)
    waitingTimeForThread: int = field(default=0, init=False)
    threadId: int = field(init=False)
    cpuId: int = field(init=False)
    numContextSwitches: int = field(default=0, init=False)
    totalContextSwitchOverhead: int = field(default=0, init=False)
    completionState: TaskCompletionState = field(init=False)
    numRetries: int = field(init=False, default=0)
    queueLengthObserved: int = field(init=False, default=0)


# The CPU class represents a core
@dataclass
class CPU:
    cpuId: int
    cpuState: CPUState = field(default=CPUState.IDLE, init=False)
    currentCpuTime: int = field(default= 0, init=False)
    totalExecutionTime: int = field(default=0, init=False)




# The user class
@dataclass
class User:
    userId: int
    thinkTime: int
    avgInterarrivalTime: int
    avgServiceTime: int
    timeoutDuration: int
    retryProb: float
    retryTime: int
    counters: Counters
    randGen: lcg
    serviceTimeDist: DistType
    task: Task = field(init=False)
    completedTasks: list[Task] = field(init=False, default_factory=list)
    userState: UserState = field(default=UserState.READY, init=False)

    def __post_init__(self):
        '''Add initial task for the user'''
        self.createTask(expon(self.avgInterarrivalTime, self.randGen))

    def createTask(self, arrivalTime: int)->None:
        '''Creates a new task for the given arrival time'''
        # get service timeout from distribution
        service_time = self.avgServiceTime if self.serviceTimeDist == DistType.CONSTANT else uniform(self.avgServiceTime, self.randGen) if self.serviceTimeDist == DistType.UNIFORM else expon(self.avgServiceTime, self.randGen)

        # 50% of the timeout is minimum and rest is exponential
        min_timeout = self.timeoutDuration // 2
        var_timeout = expon(self.timeoutDuration - min_timeout, self.randGen)
        total_timeout = min_timeout + var_timeout

        self.task = Task(self.counters.getTaskIdCounter(), self.userId, arrivalTime, service_time, total_timeout, service_time)


    def sendRequest(self) -> Task:
        '''Sends the task to server'''
        # set the user state to waiting
        self.userState = UserState.WAITING
        # print trace
        print(f"|{'ARRIVAL':15s}|{self.task.arrivalTime:<10d}|{f'USERID {self.userId}':10s}|{f'TASKID {self.task.taskId}':10s}|{f'BURST {self.task.burstTime}':10s}|")
        # return the task
        return self.task

    def droppedRequest(self) -> None: 
        '''Method to handle request drops'''
        # with probability p retry the request 
        if self.randGen.lcgrand() <= self.retryProb:
            # retry
            self.task.arrivalTime += expon(self.retryTime, self.randGen)
            self.task.numRetries += 1
            # change the state of the user
            self.userState = UserState.READY
            print(f"|{'RETRY':15s}|{self.task.arrivalTime:<10d}|{f'USERID {self.userId}':10s}|{f'TASKID {self.task.taskId}':10s}|{f'RETRYN {self.task.numRetries}':10s}|")
        else:
            print(f"|{'DROPPED':15s}|{self.task.arrivalTime:<10d}|{f'USERID {self.userId}':10s}|{f'TASKID {self.task.taskId}':10s}|{f'RETRYN {self.task.numRetries}':10s}|")
            # mark the request as dropped
            self.task.completionState = TaskCompletionState.DROPPED
            self.task.departureTime = self.task.arrivalTime
            self.completedTasks.append(self.task)

            # generate a new request
            self.createTask(self.completedTasks[-1].departureTime + self.thinkTime)
            # change userstate
            self.userState = UserState.READY


    def requestCompleted(self):
        
        print(f"|{self.task.completionState.name:15s}|{self.task.departureTime:<10d}|{f'USERID {self.userId}':10s}|{f'TASKID {self.task.taskId}':10s}|")

        # add current task to completed tasks list
        self.completedTasks.append(self.task)
        
        # think time is combination of min fixed + variable time
        think_time = round(0.8 * self.thinkTime) + expon(round(0.2 * self.thinkTime), self.randGen)

        # create a new task
        self.createTask(self.completedTasks[-1].departureTime + think_time)

        # set userState to ready
        self.userState = UserState.READY


# Class to manage list of users
@dataclass
class UserList:
    users: dict[int,User] = field(default_factory=dict)

    def add_user(self, user: User) -> None:
        '''Adds a new user to the list'''
        self.users[user.userId] = user
    
    def get_user(self, userId: int) -> User:
        '''Returns the user with the specified
           user id'''
        return self.users[userId]
    
    def get_next_sender(self) -> User:
        '''Get the next user who is ready to send the request
        return None if all users are waiting'''
        readyUsers = self.get_all_ready_users()

        if len(readyUsers) == 0:
            return None

        next_user = min(readyUsers, key=lambda x: x.task.arrivalTime)
        return next_user

    def get_all_ready_users(self) -> list[User]:
        '''Returns the list of all ready users'''
        return [ x for x in self.users.values() if x.userState == UserState.READY]

    def get_all_users(self) -> list[User]:
        '''Return list of all users'''
        return self.users.values()


# The thread class represent a thread. 
# Each thread is assigned a task
@dataclass
class Thread:
    threadId: int
    cpuId: int
    task: Task
    threadState: ThreadState = field(default=ThreadState.CREATED, init=False)

    def threadCompleted(self, currentCpuTime: int, usersList: UserList) -> int:
        '''Thread has completed execution
           Returns the associated user id'''
        # Mark thread state as done
        self.threadState = ThreadState.DONE
        # update task departure time
        self.task.departureTime = currentCpuTime
        # update total waiting time
        self.task.waitingTime = self.task.departureTime - self.task.arrivalTime - self.task.burstTime
        # turn around time
        tat = self.task.departureTime - self.task.arrivalTime

        self.task.completionState = TaskCompletionState.SUCCESS if self.task.timeoutDuration >= tat else TaskCompletionState.TIMEOUT

        # update the user
        usersList.get_user(self.task.userId).requestCompleted()
        return self.task.userId


@dataclass
class Scheduler(ABC):
    contextSwitchOverhead: int
    cpu: CPU
    maxThreadQueueSize: int
    usersList: UserList
    counters: Counters
    execThreadQueue: list[Thread] = field(default_factory=list, init=False)
    executingThreadIdx: int = field(default=-1, init=False)
    currentTimeQuanta: int = field(default=0, init=False)

    def isThreadQueueFull(self)->bool:
        '''Returns true if thread queue is full'''
        return len(self.execThreadQueue) == self.maxThreadQueueSize
    
    @abstractmethod
    def getNextTimeQuanta(self, t_task: Task) -> int:
        '''Returns time quanta'''
        pass

    def addThread(self, thread: Thread) -> bool:
        '''Adds a new thread to the thread queue
        returns true if cpu was idle before'''
        assert len(self.execThreadQueue) < self.maxThreadQueueSize

        # add thread in the last
        self.execThreadQueue.append(thread)

        cpuIdle = False

        # if this is the first thread
        if len(self.execThreadQueue) == 1:
            assert self.cpu.cpuState == CPUState.IDLE
            cpuIdle = True
            # update current time quanta
            self.currentTimeQuanta = self.getNextTimeQuanta(thread.task)
            self.executingThreadIdx = 0
            # set cpu to BUSY
            self.cpu.cpuState = CPUState.BUSY
            # set the cpu time to the arrival time
            self.cpu.currentCpuTime = max(thread.task.arrivalTime, self.cpu.currentCpuTime)
        
        # update waiting in thread time
        thread.task.waitingTimeForThread = self.cpu.currentCpuTime - thread.task.arrivalTime
        # update thread state to ready
        thread.threadState = ThreadState.READY
        return cpuIdle

    def addTaskAndCreateThread(self, t_task: Task) -> bool:
        '''Adds a task and creates thread
        returns true if cpu was idle before'''
        # create a thread and assign it to the cpu
        thread = Thread(self.counters.getThreadIdCounter(), self.cpu.cpuId, t_task)
        t_task.threadId = thread.threadId
        t_task.cpuId = self.cpu.cpuId

        # add it to the scheduler 
        cpuIdle = self.addThread(thread)

        print(f"|{'THREADCREATE':15s}|{self.cpu.currentCpuTime:<10d}|{f'CPUID {self.cpu.cpuId}':10s}|{f'TID {thread.threadId}':10s}|{f'TASKID {t_task.taskId}':10s}")

        return cpuIdle
    

    def getNextContextSwitchTime(self) -> int:
        '''Returns the context switch time of next task'''
        if self.cpu.cpuState == CPUState.IDLE:
            return -1
        return self.cpu.currentCpuTime + self.currentTimeQuanta

    def executeCurrentThread(self):
        '''Executes the current thread'''
        if self.executingThreadIdx != -1:
            # Set thread state to runnning
            self.execThreadQueue[self.executingThreadIdx].threadState = ThreadState.RUNNING
            # update cpu stats
            print(f"|{'EXECUTE':15s}|{self.cpu.currentCpuTime:<10d}|{f'CPUID {self.cpu.cpuId}':10s}|{f'TID {self.execThreadQueue[self.executingThreadIdx].threadId}':10s}|{f'FOR {self.currentTimeQuanta}':10s}|")
            self.cpu.currentCpuTime += self.currentTimeQuanta
            self.cpu.totalExecutionTime += self.currentTimeQuanta
            # update task remaining time
            self.execThreadQueue[self.executingThreadIdx].task.remainingTime -= self.currentTimeQuanta

        
    def contextSwitch(self) -> User:
        '''Gets the next task to schedule
            Returns user if its task is completed'''
        nextIdx = -1
        completedUser = None
        print(f"|{'CTXSWITCH':15s}|{self.cpu.currentCpuTime:<10d}|{f'CPUID {self.cpu.cpuId}':10s}|{f'TID {self.execThreadQueue[self.executingThreadIdx].threadId}':10s}|{f'TASKID {self.execThreadQueue[self.executingThreadIdx].task.taskId}':10s}|")

        # if the current thread has completed its execution
        # then mark it as completed and remove from queue
        if self.execThreadQueue[self.executingThreadIdx].task.remainingTime <= 0:
            
            # check if task completed
            if self.execThreadQueue[self.executingThreadIdx].task.remainingTime <= 0:
                # call compeletion handler
                uId = self.execThreadQueue[self.executingThreadIdx].threadCompleted(self.cpu.currentCpuTime, self.usersList)
                completedUser = self.usersList.get_user(uId)

            # remove thread from queue
            del self.execThreadQueue[self.executingThreadIdx]

            # since item is deleted, the current index points 
            # to the next Thread
            nextIdx = self.executingThreadIdx % len(self.execThreadQueue) if len(self.execThreadQueue) != 0 else -1

        elif len(self.execThreadQueue) == 1:
            # if there is only one task
            # then execute the same task without 
            # context switch overhead
            nextIdx = self.executingThreadIdx

        else:
            # this task is context switching out
            # increase number of context switches
            self.execThreadQueue[self.executingThreadIdx].task.numContextSwitches += 1

            # context Switch Overhead
            self.execThreadQueue[self.executingThreadIdx].task.totalContextSwitchOverhead += self.contextSwitchOverhead
            
            # Set thread state to ready
            self.execThreadQueue[self.executingThreadIdx].threadState = ThreadState.READY

            # increment index
            nextIdx = (self.executingThreadIdx + 1) % len(self.execThreadQueue)


        # if thread queue is empty then return -1
        if len(self.execThreadQueue) == 0:
            # set cpu to idle
            self.cpu.cpuState = CPUState.IDLE
            self.currentTimeQuanta = 0
            self.executingThreadIdx = -1
            return completedUser
        

        # add context switch overhead to the cpu time 
        # if there are more than 1 task
        if len(self.execThreadQueue) > 1:
            self.cpu.currentCpuTime += self.contextSwitchOverhead
            # add to the cpu execution time
            self.cpu.totalExecutionTime += self.contextSwitchOverhead

        # set the next time_quanta
        self.currentTimeQuanta = self.getNextTimeQuanta(self.execThreadQueue[nextIdx].task)

        # set the index of next scheduled thread
        self.executingThreadIdx = nextIdx
        
        # return completed user
        return completedUser


@dataclass
class RoundRobinScheduler(Scheduler):
    maxTimeQuanta: int

    def getNextTimeQuanta(self, t_task: Task) -> int:
        '''Returns time quanta'''
        return min(t_task.remainingTime, self.maxTimeQuanta)

@dataclass
class FIFOScheduler(Scheduler):

    def getNextTimeQuanta(self, t_task: Task) -> int:
        '''Returns time quanta'''
        return t_task.remainingTime

# class to represent a list of schedulers
@dataclass
class SchedulerList:
    schedulers: list[Scheduler] = field(init=False, default_factory=list)
    countsTaskAdded: dict[int,int] = field(init=False, default_factory=dict)

    def add(self,sch: Scheduler)-> None:
        '''Adds a scheduler to the list'''
        self.schedulers.append(sch)
        self.countsTaskAdded[sch.cpu.cpuId] = 0
    
    def __getitem__(self, key):
        return self.schedulers[key]

    def getAScheduler(self)->Scheduler:
        '''Returns a scheduler with cpu that fewest threads
        returns None if all scheduler is fully occupied'''

        minQSched = min(self.schedulers, key=lambda x: len(x.execThreadQueue))

        if minQSched.isThreadQueueFull():
            return None

        # if more than 1 schedulers have min thread queue count,
        # then return the one which has least countsTaskAdded value

        tScheds = [ x for x in  self.schedulers if len(x.execThreadQueue) == len(minQSched.execThreadQueue) ]
        
        targetSched = minQSched

        if len(tScheds) > 1:
            targetSched = min(tScheds, key=lambda x: self.countsTaskAdded[x.cpu.cpuId])
            
        # increase count
        self.countsTaskAdded[targetSched.cpu.cpuId] += 1
        return targetSched
    
    def nextEvent(self)->Scheduler:
        '''Returns the time of next event and associated scheduler
        None if there is no event'''
        
        busyCpus = list(filter(lambda x: x.cpu.cpuState == CPUState.BUSY, self.schedulers))

        if len(busyCpus) == 0:
            return None

        schd = min(busyCpus, key=lambda x: x.getNextContextSwitchTime())
        return schd


@dataclass
class TaskQueue:
    maxQueueLength: int
    queue: list[Task] = field(init=False, default_factory=list)

    def enqueue(self, t_task: Task) -> None:
        "Adds a task to the queue"
        # add only if the queue length is less than the max length
        assert len(self.queue) < self.maxQueueLength
        self.queue.insert(0, t_task)
    
    def dequeue(self) -> Task:
        '''Pops and returns task in FIFO order'''
        assert len(self.queue) != 0
        return self.queue.pop()
    
    def peek(self) -> Task:
        '''Returns the top task without poppping'''
        return self.queue[-1]

    def length(self) -> int:
        '''Retutns length of the queue'''
        return len(self.queue)
    
    def isFull(self) -> bool:
        '''Returns if the task queue is full'''
        return self.length() == self.maxQueueLength

@dataclass(order=True)
class Event:
    # class to represent event details
    eventTime: int
    eventType: EventType
    associatedObject: object = field(compare=False)

    def __eq__(self, __o: object) -> bool:
        '''all fields are equal'''
        return self.eventTime == __o.eventTime and self.eventType == __o.eventType and id(self.associatedObject) == id(__o.associatedObject)

@dataclass
class EventList:
    events: list[Event] = field(init=False, default_factory=list)

    def addEvent(self, event: Event) -> None:
        '''Adds a new event to the pq'''
        ## if same event is not already in the heap then add
        if event not in self.events:
            heapq.heappush(self.events, event)
    
    def getNextEvent(self) -> Event:
        '''returns the top event'''
        return heapq.heappop(self.events)
    

class EventHandlers:

    @staticmethod
    def arrivalEventHandler(eventTime: int, t_user: User, schedulers: SchedulerList, taskQueue: TaskQueue, eventsList: EventList) -> None:
        # apply arrival logic
        # check if queue is full
        t_task = t_user.sendRequest()

        if taskQueue.isFull():
            # then drop the request
            t_user.droppedRequest()
        else:
            print(f"|{'ENQUEUE':15s}|{eventTime:<10d}|{f'USERID {t_user.userId}':10s}|{f'TASKID {t_task.taskId}':10s}|{f'QLEN {taskQueue.length()}':10s}")
            # record the queue length
            t_task.queueLengthObserved = taskQueue.length()
            # add task to the queue
            taskQueue.enqueue(t_user.task)
            # TODO: to record the queue length?
        
            # check if any cpu is free
            # get a free cpu 
            freeSched = schedulers.getAScheduler()
            if freeSched != None:
                # create a deque event at the next context switch time
                eventsList.addEvent(Event(max(eventTime, freeSched.getNextContextSwitchTime()), EventType.DEQUEUE, freeSched))

    @staticmethod
    def executionEventHandler(eventTime: int, sched: Scheduler, eventsList: EventList) -> None:
        sched.executeCurrentThread()
        # Add context switch event to the events list
        eventsList.addEvent(Event(sched.cpu.currentCpuTime, EventType.CTXSWITCH, sched))

    @staticmethod
    def contextSwitchEventHandler(eventTime: int, sched: Scheduler, taskQueue: TaskQueue, schedulers: SchedulerList, usersList: UserList, eventsList: EventList) -> None:
        completedUser = sched.contextSwitch()

        # add the next Execution event 
        if sched.cpu.cpuState != CPUState.IDLE:
            eventsList.addEvent(Event(sched.cpu.currentCpuTime, EventType.EXECUTION, sched))

        # check is task_queue is not empty
        if taskQueue.length() != 0 and not sched.isThreadQueueFull():
            # check if current cpu is free
            # add dequeue event to the list
            eventsList.addEvent(Event(max(taskQueue.peek().arrivalTime,sched.cpu.currentCpuTime), EventType.DEQUEUE, sched))
        
        # if a user is completed, then add next arrival event
        if completedUser != None:
            eventsList.addEvent(Event(completedUser.task.arrivalTime, EventType.ARRIVAL, completedUser))
        
        
    
    @staticmethod
    def dequeEventHandler(eventTime: int, sched: Scheduler, taskQueue: TaskQueue, eventsList: EventList ) -> None:
        # if schduler is still not free, then don't deque
        if sched.isThreadQueueFull() or taskQueue.length() == 0:
            return
        # deque from the task queue and create a thread in scheduler
        print(f"|{'DEQUEUE':15s}|{eventTime:<10d}|{f'CPUID {sched.cpu.cpuId}':10s}|{f'TASKID {taskQueue.peek().taskId}':10s}|{f'QLEN {taskQueue.length()-1}':10s}")
        ts = taskQueue.dequeue()
        cpuIdle = sched.addTaskAndCreateThread(ts)
        if cpuIdle:
            eventsList.addEvent(Event(sched.cpu.currentCpuTime, EventType.EXECUTION, sched))

    
