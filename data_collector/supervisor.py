import keyboard

def Supervisor_loop(controller):
    while True:
        if keyboard.is_pressed('w'):
            controller.step("MoveAhead")
        elif keyboard.is_pressed('s'):
            controller.step("MoveBack")
        elif keyboard.is_pressed('a'):
            controller.step("MoveLeft")
        elif keyboard.is_pressed('d'):
            controller.step("MoveRight")
        elif keyboard.is_pressed('left'): 
            controller.step("RotateLeft")
        elif keyboard.is_pressed('right'): 
            controller.step("RotateRight")
        elif keyboard.is_pressed('up'): 
            controller.step(
                action="LookUp",
                degrees=1
            )
        elif keyboard.is_pressed('down'):
            controller.step(
                action="LookDown",
                degrees=1
            )
        elif keyboard.is_pressed('q'):
            print('Quit!')
            break