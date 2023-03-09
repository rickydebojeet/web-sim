from classes import *


class simulator:
    
    def __init__(self, num_cpus: int, max_threads: int) -> None:
        self.num_cpus = num_cpus
        self.max_threads = max_threads
        self.cpus = [ CPU(i) for i in range(num_cpus) ]
        
    
    def run_simulation(self, num_users, think_time, avg_service_time, context_switch_time, avg_interarrival_time, timeout, repeat, ctxSwOverhead,  rseed = 1, rstream = 0) -> None:
        
        task_queue = []
        schedulers = SchedulerList()
        
        # initialise the random number generator
        lcgrandst(rseed, rstream)
        lcgSetActiveStream(rstream)
        
        for i in range(num_users):
            # initilise users
            t_user = User(i, think_time, avg_interarrival_time, avg_service_time, timeout)
            GlobalVars.usersList.add_user(t_user)
        
        # create scheduler for each cpu

        for cpu in self.cpus:
            schedulers.add(RoundRobinScheduler(context_switch_time, ctxSwOverhead, cpu ))
            

        for _ in range(repeat):
            # get next user event
            nextUser = GlobalVars.usersList.get_next_sender()
            # get next scheduler event
            nextSched = schedulers.nextEvent()

            ## if both are none then usse error
            assert nextUser != None and nextSched != None

            if nextUser != None and (nextSched == None or nextUser.task.arrivalTime < nextSched.getNextContextSwitchTime()):
                # apply arrival logic
                # get a free cpu 
                freeSched = schedulers.getAScheduler()
                t_task = nextUser.sendRequest()
                if freeSched == None:
                    # there is no free cpu
                    # add to task queue
                    task_queue.insert(0, t_task)
                    # TODO: to record the queue length?
                else:
                    freeSched.addTaskAndCreateThread(t_task)
            
            else:
                # the next event is scheduler event
                nextSched.executeCurrentThread()
                nextSched.contextSwitch()

                # check is task_queue is not empty
                while len(task_queue) != 0:
                    # check if any cpu is free
                    freeSched = schedulers.getAScheduler()
                    
                    # not free cpu
                    if freeSched == None:
                        break
                        
                    # add task to scheduler
                    freeSched.addTaskAndCreateThread(task_queue.pop())
        
        


