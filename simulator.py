from classes import *
from enum import Enum

class SchedulerType(Enum):
    ROUNDROBIN = 0
    FIFO = 1

class simulator:
    
    def __init__(self, num_cpus: int, max_threads: int) -> None:
        self.num_cpus = num_cpus
        self.max_threads = max_threads
        self.cpus = [ CPU(i) for i in range(num_cpus) ]
        self.counters = Counters()
        self.usersList = UserList()
    


    def run_simulation(self, num_users, think_time, avg_service_time, context_switch_time, avg_interarrival_time, timeout, simulation_time, ctxSwOverhead, maxQueueSize, retryProb, retryTime, schedulerType: SchedulerType,  rseed = 1, rstream = 0) -> None:
        
        task_queue = TaskQueue(maxQueueSize)
        schedulers = SchedulerList()
        eventsList = EventList()

        self.simulation_time = simulation_time

        # initialise the random number generator
        lcgrandst(rseed, rstream)
        lcgSetActiveStream(rstream)
        
        for i in range(num_users):
            # initilise users
            t_user = User(i, think_time, avg_interarrival_time, avg_service_time, timeout, retryProb, retryTime, self.counters)
            self.usersList.add_user(t_user)
        
        # initially add all arrival events
        for user in self.usersList.get_all_ready_users():
            eventsList.addEvent(Event(user.task.arrivalTime, EventType.ARRIVAL, user))

        # create scheduler for each cpu
        if schedulerType == SchedulerType.ROUNDROBIN:
            for cpu in self.cpus:
                schedulers.add(RoundRobinScheduler(ctxSwOverhead, cpu, self.max_threads, self.usersList, self.counters, context_switch_time))
        elif schedulerType == SchedulerType.FIFO:
            for cpu in self.cpus:
                schedulers.add(FIFOScheduler(ctxSwOverhead, cpu, self.max_threads, self.usersList, self.counters))
        else:
            assert False

        while True:
            
            # get next event
            nextEvent = eventsList.getNextEvent()

            # check if simulation is over
            if nextEvent.eventTime > simulation_time:
                break

            # call the event handler
            if nextEvent.eventType == EventType.ARRIVAL:
                EventHandlers.arrivalEventHandler(nextEvent.eventTime, nextEvent.associatedObject, schedulers, task_queue, eventsList)
            elif nextEvent.eventType == EventType.EXECUTION:
                EventHandlers.executionEventHandler(nextEvent.eventTime, nextEvent.associatedObject, eventsList)
            elif nextEvent.eventType == EventType.CTXSWITCH:
                EventHandlers.contextSwitchEventHandler(nextEvent.eventTime, nextEvent.associatedObject, task_queue, schedulers, self.usersList, eventsList)
            elif nextEvent.eventType == EventType.DEQUEUE:
                EventHandlers.dequeEventHandler(nextEvent.eventTime, nextEvent.associatedObject, task_queue, eventsList)

                

    
    def getAvgResponseTime(self) -> float:
        '''Returns the average response time after simulation'''
        totalRequests = 0
        totalTime = 0
        
        for user in self.usersList.get_all_users():
            for com_task in user.completedTasks:
                if com_task.completionState != TaskCompletionState.DROPPED:
                    totalRequests += 1
                    totalTime += (com_task.departureTime - com_task.arrivalTime)
        
        # return the average response time
        return totalTime / totalRequests

        
    def getGoodPut(self) -> float:
        '''Get the goodput after simulation'''
        completedRequests = 0
        for user in self.usersList.get_all_users():
            for com_task in user.completedTasks:
                if com_task.completionState == TaskCompletionState.SUCCESS:
                    completedRequests += 1
        
        # divide by simulation time
        return completedRequests / self.simulation_time


    def getBadPut(self) -> float:
        '''Get the badput after simulation'''
        timedOutRequests = 0
        for user in self.usersList.get_all_users():
            for com_task in user.completedTasks:
                if com_task.completionState == TaskCompletionState.TIMEOUT:
                    timedOutRequests += 1
        
        # divide by simulation time
        return timedOutRequests / self.simulation_time
    
    def getRequestDropRate(self) -> float:
        '''Get dropped request rate after simulation'''
        droppedRequests = 0
        for user in self.usersList.get_all_users():
            for com_task in user.completedTasks:
                if com_task.completionState == TaskCompletionState.DROPPED:
                    droppedRequests += 1
        
        # divide by simulation time
        return droppedRequests / self.simulation_time

    def getCoreUtilisation(self) -> float:
        '''Returns utilisation of all cores'''
        return sum([ cpu.totalExecutionTime / cpu.currentCpuTime for cpu in self.cpus ]) / self.num_cpus

    def getAvgQueueLength(self) -> float:
        '''Returns the avg number of retries per task'''
        totalLength = 0
        totalTasks = 0
        for user in self.usersList.get_all_users():
            for com_task in user.completedTasks:
                totalLength += com_task.queueLengthObserved
                totalTasks += 1
        return totalLength / totalTasks 

    def getAvgWaitingTimeInQueue(self) -> float:
        '''Returns the average waiting time in queue'''
        totalRequests = 0
        totalTime = 0
        
        for user in self.usersList.get_all_users():
            for com_task in user.completedTasks:
                if com_task.completionState != TaskCompletionState.DROPPED:
                    totalRequests += 1
                    totalTime += com_task.waitingTimeForThread
        
        # return the average response time
        return totalTime / totalRequests



def main():
    CLOCKS_PER_SEC = 1000000
    sim = simulator(2, 50)
    sim.run_simulation(350, 10000000, 80000, 20000, 100000, 2000000, 300000000, 1000, 200, 0.5, 5000, SchedulerType.ROUNDROBIN)
    print(f'Response time: {sim.getAvgResponseTime() / CLOCKS_PER_SEC} s')
    print(f'Goodput: {sim.getGoodPut() * CLOCKS_PER_SEC} req/s')
    print(f'Badput: {sim.getBadPut() * CLOCKS_PER_SEC} req/s')
    print(f'Request Drop Rate: {sim.getRequestDropRate() * CLOCKS_PER_SEC} req/s')
    print(f'Utilisation: {sim.getCoreUtilisation()}')
    print(f'Average Queue Length: {sim.getAvgQueueLength()}')
    print(f'Avg waiting time in queue: {sim.getAvgWaitingTimeInQueue() / CLOCKS_PER_SEC}')




if __name__ == '__main__':
    main()