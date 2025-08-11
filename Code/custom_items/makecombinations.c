#include <stdlib.h>
#include <stdio.h>
#include <string.h>

char *itemsep="\n    ";
char **values;

// Generate combinations sorted by first element.
// for example:  [1,2], [1,3], [1,4], [2,3], [2,4], [3,4]

int combinations_prefix(char *prefix, int min, int max, int count)
{
  int len=strlen(prefix);
  if (count==1) {
    for (int i=min; i<=max; i++) {
      printf("%s[%s%s]", itemsep, prefix, values[i]);
      itemsep=",\n    ";
    }
  } else if (count>1) {
    sprintf(prefix+len, "%s,", values[min]);
    combinations_prefix(prefix, min+1, max, count-1);
    if (max-min+1>count) {
      prefix[len]=0;
      combinations_prefix(prefix, min+1, max, count);
    }
  }
  return 0;
}

// Generate combinations sorted by last element.
// for example:  [1,2], [1,3], [2,3], [1,4], [2,4], [3,4]

int combinations_postfix(char *postfix, int min, int max, int count)
{
  int len=strlen(postfix);
  if (count==1) {
    for (int i=min; i<=max; i++) {
      printf("%s[%s%s]", itemsep, values[i], postfix);
      itemsep=",\n    ";
    }
  } else if (count>1) {
    if (max-min+1>count) {
      postfix[len]=0;
      combinations_postfix(postfix, min, max-1, count);
    }
    // prepend the current selection.
    int vlen=strlen(values[max]);
    strncpy(postfix-vlen, values[max], vlen);
    postfix = postfix-vlen-1;
    postfix[0]=',';
    combinations_postfix(postfix, min, max-1, count-1);
  }
  return 0;
}

int main(int argc, char *argv[])
{
  //int min=atoi(argv[1]);
  //int max=atoi(argv[2]);
  int count=argc-1;
  values = argv;
  char buffer[2000];
  printf("import tensorflow as tf\n\n");
  for (int i=1; i<=count; i++) {
    printf("combi_%i_of_%i = tf.constant([", i,count);
    buffer[0]=0;
    itemsep="\n    ";
#ifdef USE_PREFIX
    combinations_prefix(buffer, 1, count, i);
#else
    combinations_postfix(buffer+1950, 1, count, i);
#endif
    printf("\n   ])\n");
  }
}
