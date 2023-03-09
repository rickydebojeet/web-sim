from dataclasses import dataclass, field
from enum import Enum
from math import log
from lcgrand import *
from typing import ClassVar

def exponf(mean: float) -> float:
    return (-mean) * log(lcgrand())

def expon(mean: int) -> int:
    return round((-mean) * log(lcgrand()))




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


# The thread class represent a thread. 
# Each thread is assigned a task
@dataclass
class Thread:
    threadId: int
    cpuId: int
    task: Task
    threadState: ThreadState = field(default=ThreadState.CREATED, init=False)

    def threadCompleted(self, currentCpuTime: int):
        '''Thread has completed execution'''
        # Mark thread state as done
        self.threadState = ThreadState.DONE
        # update task departure time
        self.task.departureTime = currentCpuTime
        # update total waiting time
        self.task.waitingTime = self.task.departureTime - self.task.arrivalTime - self.task.burstTime
        # turn around time
        tat = self.task.departureTime - self.task.arrivalTime

        self.task.completionState = TaskCompletionState.SUCCESS if self.task.timeoutDuration <= tat else TaskCompletionState.TIMEOUT

        # update the user
        GlobalVars.usersList.get_user(self.task.userId).requestCompleted()


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
    task: Task = field(init=False)
    completedTasks: list[Task] = field(init=False, default_factory=list)
    userState: UserState = field(default=UserState.READY, init=False)

    def __post_init__(self):
        '''Add initial task for the user'''
        self.createTask(expon(self.avgInterarrivalTime))

    def createTask(self, arrivalTime: int)->None:
        '''Creates a new task for the given arrival time'''
        # get service timeout from distribution
        dist_service_time = expon(self.avgServiceTime)
        self.task = Task(GlobalVars.TASKIDCOUNTER, self.userId, arrivalTime, dist_service_time, self.timeoutDuration, dist_service_time)
        GlobalVars.TASKIDCOUNTER += 1

    def sendRequest(self) -> Task:
        '''Sends the task to server'''
        # set the user state to waiting
        self.userState = UserState.WAITING
        # return the task
        return self.task

    def requestCompleted(self):
        # add current task to completed tasks list
        self.completedTasks.append(self.task)
        
        # create a new task
        self.createTask(self.completedTasks[-1].departureTime + self.thinkTime)

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
        readyUsers = list(filter(lambda x: x.userState == UserState.READY, self.users.values()))

        if len(readyUsers) == 0:
            return None

        next_user = min(readyUsers, key=lambda x: x.task.arrivalTime)
        return next_user



@dataclass
class RoundRobinScheduler:
    maxTimeQuanta: int
    contextSwitchOverhead: int
    cpu: CPU
    maxThreadQueueSize: int
    execThreadQueue: list[Thread] = field(default_factory=list, init=False)
    executingThreadIdx: int = field(default=-1)
    currentTimeQuanta: int = field(default=0, init=False)

    def isThreadQueueFull(self)->bool:
        '''Returns true if thread queue is full'''
        return len(self.execThreadQueue) == self.maxThreadQueueSize

    def addThread(self, thread: Thread) -> None:
        '''Adds a new thread to the thread queue'''
        assert len(self.execThreadQueue) < self.maxThreadQueueSize

        # add thread in the last
        self.execThreadQueue.append(thread)

        # if this is the first thread
        if len(self.execThreadQueue) == 1:
            assert self.cpu.cpuState == CPUState.IDLE
            # update current time quanta
            self.currentTimeQuanta = min(thread.task.remainingTime, self.maxTimeQuanta)
            self.executingThreadIdx = 0
            # set cpu to BUSY
            self.cpu.cpuState = CPUState.BUSY
            # set the cpu time to the arrival time
            self.cpu.currentCpuTime = max(thread.task.arrivalTime, self.cpu.currentCpuTime)
        
        # update waiting in thread time
        thread.task.waitingTimeForThread = self.cpu.currentCpuTime - thread.task.arrivalTime

    def addTaskAndCreateThread(self, t_task: Task) -> None:
        '''Adds a task and creates thread'''
        # create a thread and assign it to the cpu
        thread = Thread(GlobalVars.THREADIDCOUNTER, self.cpu.cpuId, t_task)

        # Increment thread id counter
        GlobalVars.THREADIDCOUNTER += 1
        # add it to the scheduler 
        self.addThread(thread)
    

    def getNextContextSwitchTime(self) -> int:
        '''Returns the context switch time of next task'''
        if self.cpu.cpuState == CPUState.IDLE:
            return -1
        return self.cpu.currentCpuTime + self.currentTimeQuanta

    def executeCurrentThread(self):
        '''Executes the current thread'''
        if self.executingThreadIdx != -1:
            # update cpu stats
            self.cpu.currentCpuTime += self.currentTimeQuanta
            self.cpu.totalExecutionTime += self.currentTimeQuanta

            # update task remaining time
            self.execThreadQueue[self.executingThreadIdx].task.remainingTime -= self.currentTimeQuanta



    def contextSwitch(self) -> int:
        '''Gets the next task to schedule
            Returns the current CPU time'''
        nextIdx = -1
        # if the current thread has completed its execution
        # then mark it as completed and remove from queue
        if self.execThreadQueue[self.executingThreadIdx].task.remainingTime <= 0:
            
            # check if task completed
            if self.execThreadQueue[self.executingThreadIdx].task.remainingTime <= 0:
                # call compeletion handler
                self.execThreadQueue[self.executingThreadIdx].threadCompleted(self.cpu.currentCpuTime)

            # remove thread from queue
            del self.execThreadQueue[self.executingThreadIdx]

            # since item is deleted, the current index points 
            # to the next Thread
            nextIdx = self.executingThreadIdx % len(self.execThreadQueue)

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
            return self.cpu.currentCpuTime
        

        # add context switch overhead to the cpu time 
        # if there are more than 1 task
        if len(self.execThreadQueue) > 1:
            self.cpu.currentCpuTime += self.contextSwitchOverhead
            # add to the cpu execution time
            self.cpu.totalExecutionTime += self.contextSwitchOverhead

        # set the next time_quanta
        self.currentTimeQuanta = min(self.maxTimeQuanta, self.execThreadQueue[nextIdx].task.remainingTime)

        # Set thread as running
        self.execThreadQueue[nextIdx].threadState = ThreadState.RUNNING

        # set the index of next scheduled thread
        self.executingThreadIdx = nextIdx
        
        # return the cpu time
        return self.cpu.currentCpuTime


# class to represent a list of schedulers
@dataclass
class SchedulerList:
    schedulers: list[RoundRobinScheduler] = field(init=False, default_factory=list)

    def add(self,sch: RoundRobinScheduler)-> None:
        '''Adds a scheduler to the list'''
        self.schedulers.append(sch)
    
    def __getitem__(self, key):
        return self.schedulers[key]

    def getAScheduler(self)->RoundRobinScheduler:
        '''Returns a scheduler with cpu that fewest threads
        returns None if all scheduler is fully occupied'''

        targetSched = min(self.schedulers, key=lambda x: len(x.execThreadQueue))
        
        if targetSched.isThreadQueueFull():
            return None
    
    def nextEvent(self)->RoundRobinScheduler:
        '''Returns the time of next event and associated scheduler
        None if there is no event'''
        
        busyCpus = list(filter(lambda x: x.cpu.cpuState == CPUState.BUSY, self.schedulers))

        if len(busyCpus) == 0:
            return None

        schd = min(busyCpus, key=lambda x: x.getNextContextSwitchTime())
        return schd




class GlobalVars:
    # Counter for thread ID
    TASKIDCOUNTER = 0
    THREADIDCOUNTER = 0

    usersList = UserList()