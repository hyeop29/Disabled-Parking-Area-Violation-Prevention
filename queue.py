import queue

imagequeue = queue.Queue()

imagequeue.put(1)

print(imagequeue.qsize())

imagequeue.get()


print(imagequeue.qsize())