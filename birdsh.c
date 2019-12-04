#include <sys/wait.h>		// waitpid(), wait()
#include <sys/types.h>		// waitpid(), wait()
#include <sys/stat.h>
#include <unistd.h>			// pid_t
#include <stdlib.h>			// setenv(), getenv(), free()
//#include <errno.h>
#include <stdio.h>			// printf(), fprintf(), perror(), stderr, stdin
#include <string.h>			// sizeof(), strcmp()
#include <dirent.h>			// scandir(), chdir(), getcwd()
#include <signal.h>

/*
  Function Declarations for builtin birdsh commands:
 */
int birdsh_cd(char **args);
int birdsh_help(char **args);
int birdsh_exit(char **args);
int birdsh_ls(char **args);
int birdsh_echo(char **args);
int birdsh_pwd(char **args);
int birdsh_clear(char **args);
int birdsh_setenv(char **args);
int birdsh_env(char **args);
int birdsh_rm(char **args);
int birdsh_process(char **args);
/*
  List of builtin commands, followed by their corresponding functions.
 */
char *builtin_str[] = {
	"cd",		// Half
	"help",		// <-
	"exit",		// <-
	"ls",		// Full
	"echo",		// <-
	"pwd",		// <-
	"clear",	// <-
	"setenv",	// <-
	"env",		// <-
	"rm",		// <-
	"./process" // <-
};

int (*builtin_func[]) (char **) = {
	&birdsh_cd,
	&birdsh_help,
	&birdsh_exit,
	&birdsh_ls,
	&birdsh_echo,
	&birdsh_pwd,
	&birdsh_clear,
	&birdsh_setenv,
	&birdsh_env,
	&birdsh_rm,
	&birdsh_process
};

int birdsh_num_builtins() {
	return sizeof(builtin_str) / sizeof(char *);
}

void sigint_handler(int signo) {
	printf("\nCaught SIGINT\n");
}


/*
  Builtin function implementations.
*/

//int chmod(const char *pathname, mode_t mode);


int birdsh_process(char **args)
{
	pid_t pid;
	int status;

	int i, j, inShell = 0;

	if (args[1] == NULL) {				// An empty command was entered.
		return 1;
	}

	for (j = 1; args[j] != NULL; j++) 	// Replaced 0th element of args by 1st element, and so on.
	{									// As 0th element is ./process
		args[j - 1] = args[j];			
	}
	args[j - 1] = args[j];

	pid = fork();
	if (pid == 0) {
		// Child process
		if (execvp(args[0], args) == -1) {		// Replaces the current running program with a new one.
			perror("birdsh");					// If returns anything, its an error
		}
		exit(EXIT_FAILURE);
	} else if (pid < 0) {
		// Error forking
		perror("birdsh");
	} else {
		// Parent process
		do {
			waitpid(pid, &status, WUNTRACED);	// Parent waits for Child to finish execution.
		} while (!WIFEXITED(status) && !WIFSIGNALED(status));	// !(Returns a nonzero value if the child process terminated normally with exit AND child process received a signal that was not handled).
	}
	return 1;
}


int birdsh_clear(char **args)
{
	int i;
	if (args[1] == NULL) {
		for (i = 0; i < 50; i++)
			printf("\n");
	}
	else
	{
		fprintf(stderr, "birdsh: expected argument to \"clear\"\n");
	}
}

int birdsh_rm(char **args)
{
	if (args[1] != NULL)
		remove(args[1]);
	return 1;
}

int birdsh_setenv(char **args)
{
	setenv(args[1], args[2], 1);
	return 1;
}

int birdsh_env(char **args)
{
	printf("%s = %s\n", args[1], getenv(args[1]));
	return 1;
}

int birdsh_pwd(char **args)
{
	char cwd[1024];
	if (args[1] == NULL) {
		getcwd(cwd, sizeof(cwd));		// Get current working directory
		printf("Current working dir: %s\n", cwd);
	}
	else {
		fprintf(stderr, "birdsh: expected argument to \"pwd\"\n");
	}
	return 1;
}

/**
   @brief Bultin command: change directory.
   @param args List of args.  args[0] is "cd".  args[1] is the directory.
   @return Always returns 1, to continue executing.
 */
int birdsh_cd(char **args)
{
	if (args[1] == NULL) {
		fprintf(stderr, "birdsh: expected argument to \"cd\"\n");
	} else {
		if (chdir(args[1]) != 0) {		// Change working directory
			perror("birdsh");
		}
	}
	return 1;
}

/**
   @brief Builtin command: print help.
   @param args List of args.  Not examined.
   @return Always returns 1, to continue executing.
 */
int birdsh_help(char **args)
{
	int i;
	printf("----------(.)> Bird Shell----------\n");
	printf("Type program names and arguments, and hit enter.\n");
	printf("The following are built in:\n");

	for (i = 0; i < birdsh_num_builtins(); i++) {
		printf("  %s\n", builtin_str[i]);			// Print built-in commands
	}
	return 1;
}

int birdsh_echo(char **args)
{
	int i;
	for (i = 1; (args[i] != NULL); i++)
	{
		printf("%s ", args[i]);		// Printing input from echo. Can print 128 words.
	}
	printf("\n");
	return 1;
}

/**
   @brief Builtin command: exit.
   @param args List of args.  Not examined.
   @return Always returns 0, to terminate execution.
 */
int birdsh_exit(char **args)
{
	return 0;		// Exit
}

/**
  @brief Launch a program and wait for it to terminate.
  @param args Null terminated list of arguments (including program).
  @return Always returns 1, to continue execution.
 */
int birdsh_ls(char **args)
{
	struct dirent **namelist;
	int n;		// Stores quantity of items in directory

	if (args[1] == NULL)
	{
		n = scandir(".", &namelist, NULL, alphasort);	// Scan current directory for matching entries, no compare, sort
	}
	else
	{
		n = scandir(args[1], &namelist, NULL, alphasort);	// scan a directory for matching entries
	}
	if (n < 0)
	{
		perror("scandir");
		exit(EXIT_FAILURE);
	}
	else
	{	// Deallocate the memory and print the file / directory names.
		while (n--)
		{
			printf("%s\n", namelist[n]->d_name);
			free(namelist[n]);
		}
		free(namelist);
	}
	return 1;
}


int birdsh_launch(char **args)
{
	pid_t pid;
	int status;

	pid = fork();
	if (pid == 0) {
		// Child process
		if (execvp(args[0], args) == -1) {		// Replaces the current running program with a new one.
			perror("birdsh");					// If returns anything, its an error
		}
		exit(EXIT_FAILURE);
	} else if (pid < 0) {
		// Error forking
		perror("birdsh");
	} else {
		// Parent process
		do {
			waitpid(pid, &status, WUNTRACED);	// Parent waits for Child to finish execution.
		} while (!WIFEXITED(status) && !WIFSIGNALED(status));	// !(Returns a nonzero value if the child process terminated normally with exit AND child process received a signal that was not handled).
	}

	return 1;
}

/**
   @brief Execute birdsh built-in or launch program.
   @param args Null terminated list of arguments.
   @return 1 if the birdsh should continue running, 0 if it should terminate
 */
int birdsh_execute(char **args)
{
	int i, inShell = 0;

	if (args[0] == NULL) {
		// An empty command was entered.
		return 1;
	}

	for (i = 0; i < birdsh_num_builtins(); i++) {
		if (strcmp(args[0], builtin_str[i]) == 0) {
			return (*builtin_func[i])(args);
			inShell = 1;
		}
	}
	if (inShell == 1)
		return birdsh_launch(args);
	else
		printf("Not in the birdSh\n");
}

#define birdsh_RL_BUFSIZE 1024
/**
   @brief Read a line of input from stdin.
   @return The line from stdin.
 */
char *birdsh_read_line(void)
{
	char *line = NULL;
	ssize_t bufsize = 0; // have getline allocate a buffer for us
	getline(&line, &bufsize, stdin);	// ssize_t getline(char **lineptr, size_t *n, FILE *stream);
	return line;
}


#define birdsh_TOK_BUFSIZE 64
#define birdsh_TOK_DELIM " \t\r\n\a"	// Token separator / delimiter

/**
   @brief Split a line into tokens (very naively).
   @param line The line.
   @return Null-terminated array of tokens.
 */
char **birdsh_split_line(char *line)
{
	int bufsize = birdsh_TOK_BUFSIZE, position = 0;
	char **tokens = malloc(bufsize * sizeof(char*));
	char *token, **tokens_backup;

	if (!tokens) {
		fprintf(stderr, "birdsh: allocation error\n");
		exit(EXIT_FAILURE);
	}

	token = strtok(line, birdsh_TOK_DELIM); // Breaks string line into a series of tokens using the delimiter.
	while (token != NULL) {
		tokens[position] = token;
		position++;

		// If we have exceeded the buffer, reallocate.
		if (position >= bufsize) {
			bufsize += birdsh_TOK_BUFSIZE;	// Increment bufsize by 1024 if (buffer exceeds)
			tokens_backup = tokens;
			tokens = realloc(tokens, bufsize * sizeof(char*));	// Attempts to resize the memory block pointed to by ptr. void *realloc(void *ptr, size_t size);
			if (!tokens) {
				free(tokens_backup);
				fprintf(stderr, "birdsh: allocation error\n");
				exit(EXIT_FAILURE);
			}
		}

		token = strtok(NULL, birdsh_TOK_DELIM);
	}
	tokens[position] = NULL;
	return tokens;
}

/**
   @brief Loop getting input and executing it.
 */
void birdsh_loop(void)
{
	char *line;
	char **args;
	int status;

	do {
		printf("(.)> ");				// Bird
		line = birdsh_read_line();		// Read
		args = birdsh_split_line(line);	// Parse
		status = birdsh_execute(args);	// Execute. Status variable determines when to exit.

		free(line);		// Deallocates the memory
		free(args);		// Deallocates the memory
	} while (status);
}

/**
   @brief Main entry point.
   @param argc Argument count.
   @param argv Argument vector.
   @return status code
 */
int main(int argc, char **argv)
{
	signal(SIGINT, sigint_handler);
	// Run command loop.
	birdsh_loop();

	// Perform cleanup.
	return EXIT_SUCCESS;
}
