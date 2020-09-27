
label_to_str = {
    #0: 'unknown',
    1: 'vehicle',
    2: 'pedestrian',
    3: 'sign',
    4: 'cyclist'
}

def label_to_box(l):
    return [
        l.box.center_x, 
        l.box.center_y, 
        l.box.center_z, 
        l.box.width, 
        l.box.length, 
        l.box.height, 
        l.box.heading,
        l.type,
    ]

def print_table_head():
    print(f"{'frame':<8}{'all':<20}", end='')
    for label in label_to_str.keys():
        print(f"{label_to_str[label]:<20}", end='')
    print()
