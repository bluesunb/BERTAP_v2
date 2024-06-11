# --push: push to remote
# --pull: pull from remote

LOCAL_PATH=$(pwd)
REMOTE_PATH="/home/user/PycharmProjects/tmp/BERTAP_v2"

# if --push
if [ "$1" == "--push" ]; then
    scp -rP 22 $LOCAL_PATH/* user@166.104.112.89:$REMOTE_PATH
fi

# if --pull
if [ "$1" == "--pull" ]; then
    scp -rP 22 user@16.104.112.89:$REMOTE_PATH/* $LOCAL_PATH
fi