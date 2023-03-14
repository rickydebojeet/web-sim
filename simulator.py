from classes import *


class simulator:
    
    def __init__(self, num_cpus: int, max_threads: int) -> None:
        self.num_cpus = num_cpus
        self.max_threads = max_threads
        self.cpus = [ CPU(i) for i in range(num_cpus) ]
        self.counters = Counters()
        self.usersList = UserList()
    


    def run_simulation(self, num_users, think_time, avg_service_time, context_switch_time, avg_interarrival_time, timeout, simulation_time, ctxSwOverhead, maxQueueSize, retryProb, retryTime, rseed = 1, rstream = 0) -> None:
        
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
        for cpu in self.cpus:
            schedulers.add(RoundRobinScheduler(context_switch_time, ctxSwOverhead, cpu, self.max_threads, self.usersList, self.counters ))
            

        while True:
            
            # get next event
            nextEvent = eventsList.getNextEvent()

            # check if simulation is over
            if nextEvent.eventTime > simulation_time:
                break

            # call the event handler
            if nextEvent.eventType == EventType.ARRIVAL:
                EventHandlers.arrivalEventHandler(nextEvent.associatedObject, schedulers, task_queue, eventsList)
            elif nextEvent.eventType == EventType.EXECUTION:
                EventHandlers.executionEventHandler(nextEvent.associatedObject, eventsList)
            elif nextEvent.eventType == EventType.CTXSWITCH:
                EventHandlers.contextSwitchEventHandler(nextEvent.associatedObject, task_queue, schedulers, self.usersList, eventsList)
            elif nextEvent.eventType == EventType.DEQUEUE:
                EventHandlers.dequeEventHandler(nextEvent.associatedObject, task_queue, eventsList)

                

    
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



def main():
    CLOCKS_PER_SEC = 1000000
    sim = simulator(1, 200)
    sim.run_simulation(15, 10000000, 80000, 20000, 100000, 2000000, 300000000, 1000, 10, 0.5, 500000)
    print(f'Response time: {sim.getAvgResponseTime() / CLOCKS_PER_SEC} s')
    print(f'Goodput: {sim.getGoodPut() * CLOCKS_PER_SEC} req/s')
    print(f'Badput: {sim.getBadPut() * CLOCKS_PER_SEC} req/s')
    print(f'Request Drop Rate: {sim.getRequestDropRate() * CLOCKS_PER_SEC} req/s')
    print(f'Utilisation: {sim.getCoreUtilisation()}')


if __name__ == '__main__':
    main()