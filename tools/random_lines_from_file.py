import sys
import random

if __name__ == "__main__":
    input_file = sys.argv[1]
    num_items = int(sys.argv[2])

    lines_cnt = 0
    for line in open(input_file, mode="r"):
        lines_cnt += 1

    lines_to_take = set(random.sample(range(lines_cnt), num_items))
    lines_list = []
    for line_id, line in enumerate(open(input_file, mode="r")):
        if line_id in lines_to_take:
            lines_list.append(line)

    random.shuffle(lines_list)
    with open(input_file+".rnd{0}".format(num_items), mode="w") as out:
        for line in lines_list:
            out.write(line)



