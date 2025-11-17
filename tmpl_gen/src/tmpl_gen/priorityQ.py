#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 18:54:49 2023

@author: icardei

Changes:
    11/14/2024: added PrioQueue.pop2
"""

import heapq

class PrioQueue(object):
    """
        Priority queue using < operator on elements for ordering.
        Pops in ascending key order.
        No conserving property, i.e. no FIFO order for the same key.
    """
    def __init__(self):
        self.queue = []

    def __len__(self):
        return len(self.queue)

    def __str__(self):
        #return ", ".join(self.queue)
        return str(self.queue)
    
    def __repr__(self):
        return str(self)

    def size(self):
        return len(self)


    def isEmpty(self):
        return len(self.queue) == 0

    def push(self, item):
        heapq.heappush(self.queue, item)

    def pop(self):
        return heapq.heappop(self.queue)



class PrioQueue2(object):
    """
        Priority queue using < operator on elements for ordering.
        Pops in ascendig key order.
        Has conserving property, i.e. FIFO order for the same key.
        User supplies the priority.
    """
    def __init__(self):
        self.queue = []
        self.index = 0

    def isEmpty(self):
        return self.size() == 0

    def size(self):
        return len(self.queue)

    def push(self, item, priority):
        heapq.heappush(self.queue, (priority, self.index, item))
        self.index += 1        
        #print("pushed " + str(item))

    def pop(self):
        """
        Remove lowest priority numbered entry.
        """
        return heapq.heappop(self.queue)[-1]

    def pop2(self):
        """
        Remove lowest priority numbered entry.
        Returns (priority, item)
        """
        (p, idx, item) = heapq.heappop(self.queue)
        return (p, item)

