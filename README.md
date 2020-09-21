# Bird Shell

Bird Shell is a simple implementation of a shell in C. Implemented following commands:

- cd
- help
- exit
- ls
- echo
- pwd
- clear
- setenv
- env
- rm
- ./process

Limitations:

- Commands must be on a single line.
- Arguments must be separated by whitespace.
- No quoting arguments or escaping whitespace.
- No piping or redirection.

## Make
```sh
gcc -static -o birdsh birdsh.c
```

## Run
```sh
./birdsh
```

## Built-in Commands:
01. cd
02. help
03. exit
04. ls
05. echo
06. pwd
07. clear
08. setenv
09. env
10. rm
11. ./process
