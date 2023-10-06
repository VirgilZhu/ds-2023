class Node(object):
    def __init__(self, elem):
        self.elem = elem
        self.next = None

class SingleLinkedList:
    def is_empty(self):
        return self._head == None

    def __init__(self, node=None):
        if node != None:
            headNode = Node(node)
            self._head = headNode
        else:
            self._head = node

    def length(self):
        count = 0
        current = self._head
        while current != None:
            count += 1
            current = current.next
            return count

    def add(self, item):
        node = Node(item)
        node.next = self._head
        self._head = node

    def append(self, item):
        node = Node(item)
        if self.is_empty():
            self._head = node
        else:
            current = self._head
            while current.next != None:
                current = current.next
            current.next = node

    def insert(self, pos, item):
        if pos <= 0:
            self.add(item)
        elif pos > (self.length() - 1):
            self.append(item)
        else:
            node = Node(item)
            count = 0
            preNode = self._head
            while count < (pos - 1):
                count += 1
                preNode = preNode.next
            node.next = preNode.next
            preNode.next = node

    def search(self, item):
        current = self._head
        while current != None:
            if current.elem == item:
                return True
            current = current.next
        return False

    def remove(self, item):
        current = self._head
        preNode = None
        while current != None:
            if current.elem == item:
                if preNode == None:
                    self._head = current.next
                else:
                    preNode.next = current.next
            break
        else:
            preNode = current
            current = current.next

    def travel(self):
        curNode = self._head
        while curNode != None:
             print(curNode.elem, end='\t')
             curNode = curNode.next
        print("  ")

if __name__ == "__main__":
    singleLinkedList = SingleLinkedList(30)
    print("是否为空链表：", singleLinkedList.is_empty())
    print("链表长度为：", singleLinkedList.length())
    print("----遍历链表----")
    singleLinkedList.travel()
    print("-----查找-----\n", singleLinkedList.search(30))
    print("-----头部插入-----")
    singleLinkedList.add(1)
    singleLinkedList.add(2)
    singleLinkedList.add(3)
    singleLinkedList.travel()
    print("-----尾部追加-----")
    singleLinkedList.append(10)
    singleLinkedList.append(20)
    singleLinkedList.travel()
    print("-----指定位置插入-----")
    singleLinkedList.insert(-1, 100)
    singleLinkedList.travel()
    print("-----删除节点-----")
    singleLinkedList.remove(100)
    singleLinkedList.travel()