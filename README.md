# Documentation

## How to start project

If you use VS Code, than you don't need start container. VS Code open folder in container for you.
Otherwise to start "dl-py" container and exec bash into container:

```
docker run --name=dl-py -it -d --restart always -v /path/to/DL-py:/home/DL-py tommasosacramone/dl-py
docker exec -w /home/DL-py -it dl-py bash 
```

To restart "dl-py" container:

```
docker start dl-py
docker exec -w /home/DL-py -it dl-py bash 
```
