#include <stdio.h>

typedef struct
{
    int data[5];
} vector;

int main(int argc, char *argv[])
{
    printf("%d %s\n", argc, argv[0]);
    vector a;
    a.data = {1,2,3,4,5};
    printf("%d\n", a.data);

    return 0;
}