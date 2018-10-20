#!/bin/bash
nvidia-smi | grep python | awk '{print $3}' | xargs kill -9
