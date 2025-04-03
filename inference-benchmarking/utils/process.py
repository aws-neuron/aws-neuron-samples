import errno
import os
import signal
import socket
import time

import psutil
import requests


def kill_process_and_children(pid):
    try:
        print(f"Terminating process with pid {pid}")
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)

        # Send SIGTERM to parent and children
        for process in [parent] + children:
            print(f"Sending SIGTERM to process with PID: {process.pid}")
            process.send_signal(signal.SIGTERM)

        # Wait for processes to terminate
        gone, alive = psutil.wait_procs([parent] + children, timeout=30)

        # If any processes are still alive, send SIGKILL
        for process in alive:
            print(f"Process with PID: {process.pid} did not terminate, sending SIGKILL")
            process.send_signal(signal.SIGKILL)

        print(
            f"Successfully terminated process with PID: {pid} and its children: {[child.pid for child in children]}"
        )
    except Exception as e:
        print(f"Failed to terminate process with PID: {pid}. Exception {e}")


def check_server_terminated(url, retries=2, delay=30):
    print("Checking if server is in terminated state")
    for i in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print(
                    f"Attempt {i + 1}/{retries}: Server is not terminated yet. Re-checking in {delay} seconds..."
                )
        except requests.ConnectionError:
            print("Server is in terminated state.")
            return True
        time.sleep(delay)

    print("Server did not respond within the retry limit.")
    return False


def is_port_available(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("localhost", port))
        return True
    except socket.error as e:
        if e.errno == errno.EADDRINUSE:
            return False
        else:
            # Handle other potential errors
            print(f"Unexpected error checking port {port}: {e}")
            return False
    finally:
        sock.close()


def find_free_port(start_port=8000, max_port=65535):
    for port in range(start_port, max_port):
        if is_port_available(port):
            return port
    raise RuntimeError("Unable to find a free port")
