import asyncio
import socketio
import time
from model.gps_inpaint import *

sio = socketio.AsyncClient()

num = 0
running = True  # Flag to control the process loop

def callGPS(main_path,dir_list_input,dir_output):
    # create_fol(dir_output)
    # create_fol(dir_list_input+'Top')
    target = []
    # source = []
    # create_fol(main_path+'output')
    # create_fol(main_path+'Top')
    # dir_list_input = glob.glob(main_path+'*'+'.jpg')
    for i in range(len(dir_list_input)):
        name_file = (dir_list_input[i].split('\\'))[1]
        num_file = ((name_file.split('_'))[1])[:-4]
        target.append(int(num_file))

    for i,coor in enumerate(target):
        main_gps(main_path,coor)
    #     jobs[task_id].progress = i+1
    # jobs[task_id].status = "complete"

@sio.event
async def connect():
    print('connection established')
    # await message('Hello from client')

@sio.event
async def message(data):
    print('message received with ', data)
    # await sio.emit('my_response', 'ok')
    # await sio.sleep(10)
    # await sio.emit('my_response', 'sec')

@sio.event
async def disconnect():
    print('disconnected from server')


@sio.on('hello')
async def on_message(data):
    print('I received "',data,'"')
    await sio.emit('my_response', 'ok')

# @sio.on('run_progress')
# async def show_progress(data):
#     print('client show process')
#     await sio.emit('run_progress',num)

# Listening for stop_process event from server
@sio.event
def stop_process():
    print("Received stop command from server. Stopping process, but staying connected.")
    global running
    running = False  # Set the running flag to False to exit the loop

@sio.on('pre_run')
async def prepare_coor(data):
    print(data)
    res_coor = read_coor(data)
    await sio.emit('run_progress', res_coor)

@sio.on('run')
async def process(data):
    global running
    running = True

    dir_list_input = data["input"]
    main_path = data["main_path"]
    save_path = data["save_fol"]
    target = []
    # source = []
    # create_fol(main_path+'output')
    # create_fol(main_path+'Top')
    # dir_list_input = glob.glob(main_path+'*'+'.jpg')
    # print(dir_list_input[0])
    for i in range(len(dir_list_input)):
        name_file = (dir_list_input[i].split('/'))[-1]
        num_file = ((name_file.split('_'))[1])[:-4]
        target.append(int(num_file))
        # break

    # print(type(target[0]),' ',target[0])
    await sio.emit('run_progress', 'running')
    for i,coor in enumerate(target):
        if(running):
            # time.sleep(5)
            print('-------',coor,'-------')
            main_gps(main_path,save_path,coor)
            await sio.emit('run_progress', i+1)
            await sio.sleep(1)  # Use sio.sleep to avoid blocking the event loop
        else:
            break
    print(running)
    if(running):
        await sio.emit('run_progress', 'wait')
        await sio.emit('clientResponse','finish')
    else:
        await sio.emit('run_progress', 'stop')


    # print(len(data["input"]),' ',data["input"][0],' ',data["save_fol"])
    # i = 0
    # while running:
    #     # print(data)
    #     # if(data != "end"):
    #     time.sleep(3)
    #     print(i)
    #     await sio.emit('run_progress', i)
    #     await sio.sleep(0)
    #     if(i<10):
    #         i = i+1
    #     else:
    #         print('finish')
    #         await sio.emit('my_response', 'finish')
    #         break
    #     # else:
    #     #     print('break')
    #     #     # exit()
    #     #     break
    #     # print("Running process...",i)
    #     # await sio.emit('run_progress', i)
    #     # await sio.sleep(1)  # Use sio.sleep to avoid blocking the event loop
    #     # i = i + 1
    # if(running):
    #     await sio.emit('run_progress', 'complete')
    #     await sio.emit('clientResponse','finish')
    # else:
    #     await sio.emit('run_progress', 'stop')
    # print('out of loop')

    # await sio.emit('my_response', 'already get data')

async def main():
    await sio.connect('http://localhost:5000',namespaces=['/'])
    # await sio.sleep(30)
    await sio.wait()
    # connected = False
    # while not connected:
    #     try:
    #         sio.connect("http://127.0.0.1:3008")
    #         print("Socket established")
    #         connected = True
    #     except Exception as ex:
    #         print("Failed to establish initial connnection to server:", type(ex).__name__)
    #         time.sleep(2)

if __name__ == '__main__':
    # asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())