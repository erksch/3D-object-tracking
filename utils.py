labels = [1, 2, 3, 4]

label_to_str = {
    #0: 'unknown',
    1: 'vehicle',
    2: 'pedestrian',
    3: 'sign',
    4: 'cyclist'
}

def label_to_box(l, id):
    return [
        l.box.center_x, 
        l.box.center_y, 
        l.box.center_z, 
        l.box.width, 
        l.box.length, 
        l.box.height, 
        l.box.heading,
        l.type,
        id,
    ]
    
def print_chart(objects, n):
    for label in ['vehicle', 'pedestrian']:
        c = 0
        for id in reversed(list(objects.keys())):
            entries = objects[id]
            type_str = label_to_str[entries[list(entries.keys())[0]][7]]
            if type_str != label: continue
            c += 1
            r = range(0, n)
            COLOR = '\033[91m' if type_str == 'vehicle' else '\033[94m'
            ENDC = '\033[0m'
            print(f"{COLOR}{id:3} {type_str:10} {''.join(['▮' if i in entries else '-' for i in r])}{ENDC}")
        print(f"Label {c}")

def print_mappings(real_objects, hypothesis, mappings, n):
    GREEN = '\033[92m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    ENDC = '\033[0m'
    frame_range = range(0, n)
    mme = {}
    misses = {}
    false_positives = {}
    mismatch_memory = {}


    for label in ['vehicle', 'pedestrian']:
        mme[label] = 0
        misses[label] = 0
        false_positives[label] = 0

        for o_idx in reversed(list(real_objects.keys())):
            mismatch_memory[o_idx] = []
            entries = real_objects[o_idx]
            type_str = label_to_str[entries[list(entries.keys())[0]][7]]
            if type_str != label: continue
            print(f"{o_idx:3} {type_str:10}", end='')
            for frame in frame_range:
                if frame not in entries:
                    print(f"-", end='')
                elif o_idx not in mappings[frame]:
                    print(f"{RED}▮{ENDC}", end='')
                    misses[label] += 1
                elif frame > 0 :
                    found_mismatch = False
                    for prev_frame in reversed(range(frame)):
                        if prev_frame in mismatch_memory[o_idx]: break
                        if o_idx in mappings[prev_frame] and mappings[prev_frame][o_idx] != mappings[frame][o_idx]:
                            print(f"{BLUE}▮{ENDC}", end='')
                            mme[label] += 1
                            found_mismatch = True
                            mismatch_memory[o_idx].append(prev_frame)
                            break
                    if not found_mismatch:
                        print(f'▯', end='')
                else:
                    print(f'▯', end='')
            print()

    print('Misses')
    print(misses)
    print('mme')
    print(mme)
    print('false_positives')
    print(false_positives)
