import libtmux

datasets = [1134, 1495, 41147, 316, 1085, 1046, 1111, 55, 1116, 448, 1458, 162, 1101, 1561, 1061, 1506, 1235, 4135, 151, 51, 41138, 40645, 1510, 1158, 312, 38, 52, 1216, 41007, 1130]


parallelism = 8
server = libtmux.Server()

data_id = 0

while True:
    session = server.new_session(session_name="data" + str(datasets[data_id]), kill_session=True, attach=False)
    session.attached_pane.send_keys('hello')
    session.attached_pane.send_keys('hello')
    session.kill_session()