def do_work(col, data):
    """Toy sample work"""
    return data.result().iloc[:, col].mean()

# Helper for size anaylsis
def get_(size):
    # https://stackoverflow.com/a/49361727/2601448
    power = 2**10
    n = 0
    Dic_powerN = {0 : '', 1: 'kilo', 2: 'mega', 3: 'giga', 4: 'tera'}
    while size > power:
        size /=  power
        n += 1
    return "{} {}".format(size, Dic_powerN[n]+'bytes')
