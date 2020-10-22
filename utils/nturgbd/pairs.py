class get_pairs():
    def __init__(self):
        self.head = [(2, 3), (2, 20), (20, 4), (20, 8)]
        self.lefthand = [(4, 5), (5, 6), (6, 7), (7, 22), (22, 21)]
        self.righthand = [(8, 9), (9, 10), (10, 11), (11, 24), (24, 23)]
        self.torso = [(20, 4), (20, 8), (20, 1), (1, 0), (0, 12), (0, 16)]
        self.leftleg = [(0, 12), (12, 13), (13, 14), (14, 15)]
        self.rightleg = [(0, 16), (16, 17), (17, 18), (18, 19)]
        self.parts_connection = [(9, 1), (5, 1), (13, 1), (17, 1), (2, 1), (9, 0), (5, 0), (13, 0), (10, 1), (10, 0), (6, 1),
                            (6, 0)]
        self.total_collection = set(self.head + self.lefthand + self.righthand + self.torso + self.leftleg + self.rightleg + self.parts_connection)

if __name__ == '__main__':
    pairs = get_pairs()
    print(pairs.total_collection)
