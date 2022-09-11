#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#define MAX 1000
void admin_section();
int main();
// Defining a function which deletes item from menu//
int del_menu()
{
	system("color 6E");
	int lno,ent, ctr = 0;
	char ch;
	FILE *fptr1, *fptr2;
	char fname[MAX] = "menu.txt";
	char str[MAX], temp[] = "temp.txt";
	fptr1 = fopen(fname, "r");
	fptr2 = fopen(temp, "w");
	printf("\nInput the line you want to remove : ");
	scanf("%s", &lno);
	lno++;
	while (!feof(fptr1))
	{
		strcpy(str, "\0");
		fgets(str, MAX, fptr1);
		if (!feof(fptr1))
		{
			ctr++;
			if (ctr != lno)
			{
				fprintf(fptr2, "%s", str);
			}
		}
	}
	fclose(fptr1);
	fclose(fptr2);
	remove(fname);
	rename(temp, fname);
	fptr1 = fopen(fname, "r");
	ch = fgetc(fptr1);
	printf(" Now the content of the file %s is : \n", fname);
	while (ch != EOF)
	{
		printf("%c", ch);
		ch = fgetc(fptr1);
	}
	fclose(fptr1);
	return 0;
}
// Defining a function which will print a line of stars//
void print_line()
{
	int i = 1;
	printf("\t\t");
	for (i = 1; i <= 80; i++)
		printf("*");
	printf("\n");
}
// Defining a function which will display menu to customer//
void file_menucustomer()
{
	system("color 6E");
	char c;
	FILE *fp1;
	fp1 = fopen("menu.txt", "r");
	while (c != EOF)
	{
		c = fgetc(fp1);
		putchar(c);
	}
	fclose(fp1);
}
// Defining a function which will display slaes to admin from the sales file//
void sales()
{
	system("color 6E");
	char c;
	FILE *fptr1;
	fptr1 = fopen("sales.txt", "r");
	while (c != EOF)
	{
		c = fgetc(fptr1);
		putchar(c);
	}
	fclose(fptr1);
}
// Defining a function which will add new item to the menu//
void file_menuadmin()
{
	system("color 6E");
	char c, item[200];
	FILE *fp1;
	fp1 = fopen("menu.txt", "a");
	printf("\nEnter the new item: ");
	gets(item);
	fprintf(fp1, "\n\t\t\t");
	fprintf(fp1, item);
	fclose(fp1);
}
// Defining a function which will display admin section//
void admin_section()
{
	system("color 6E");
	int ent;
	char uname[25], pwd[25], un[25], pd[25], item[200];
	FILE *fp, *fp1;
	_flushall();
	fp = fopen("pass.txt", "r");
	fscanf(fp, "%s", uname);
	fscanf(fp, "%s", pwd);
	printf("Enter Username: ");
	gets(un);
	printf("\nEnter Password: ");
	gets(pd);
	if ((strcmp(uname, un) == 0) && (strcmp(pwd, pd) == 0))
	{
		printf("\nLogin Successful!!\n\n");

		print_line();
		printf("\n\t\t\t\t\t\tADMIN SECTION\n\n");
		print_line();
		printf("\n\t\t 1. View Total Sales\n");
		printf("\t\t 2. Add New Items In The Order Menu\n");
		printf("\t\t 3. Delete Items From The Order Menu\n");
		printf("\t\t 4. Display Order Menu\n");
		printf("\t\t 5. Back To Main Menu\n");
		printf("\t\t\nENTER YOUR CHOICE--> ");
		scanf("%d", &ent);
		printf("\n");
		switch (ent)
		{
		case 1:
		{
			print_line();
			printf("\t\t\t\t\tTotal Bills Till Date\n");
			print_line();
			sales();
			printf("\n\nEnter 5 to Return to the Main Menu: ");
			scanf("%d", &ent);
			if (ent == 5)
			{
				printf("\n");
				main();
				break;
			}
		}
		case 2:
		{
			_flushall();
			file_menucustomer();
			file_menuadmin();
			printf("\nNew item added successfully");
			printf("\n\nEnter 5 to Return to the Main Menu: ");
			scanf("%d", &ent);
			if (ent == 5)
			{
				printf("\n");
				main();
			}
		}
		case 3:
		{
			file_menucustomer();
			del_menu();
			printf("\n\nEnter 5 to Return to the Main Menu: ");
			scanf("%d", &ent);
			if (ent == 5)
			{
				printf("\n");
				main();
			}
			break;
		}
		case 4:
		{
			file_menucustomer();
			printf("\n\nEnter 5 to Return to the Main Menu: ");
			scanf("%d", &ent);
			if (ent == 5)
			{
				printf("\n");
				main();
			}
			break;
		}
		case 5:
		{
			main();
			break;
		}
		}
	}
	else
	{
		printf("\nIncorrect Username or Password!!\n\n");
	}
}
// Title page of the system//
int main()
{

	system("color 6E");
	int ch, ent, m1, count = 0;
	char mob[12];
	char m1_1, name[20], add[50];
	int arr[] = {80, 60, 70, 80, 80, 90, 100, 250, 250, 250, 200, 300, 40, 250,120};
	int sum = 0;
	print_line();
	printf("\t\t                     WELCOME TO RESTAURANT MANAGEMENT SYSTEM\n");
	print_line();
	printf("\t\t1. ADMIN SECTION -->\n");
	printf("\t\t2. CUSTOMER SECTION -->\n");
	printf("\t\t3. EXIT -->\n");
	printf("\t\t\nCHOOSE FROM THE ABOVE OPTIONS: ");
	scanf("%d", &ch);
	printf("\n");
	switch (ch)
	{
	case 1:
		admin_section();
		break;

	case 2:
	{
		print_line();
		printf("\t\t\t\t\tCUSTOMER SECTION\n");
		print_line();
		_flushall();
		FILE *fptr1;
		fptr1 = fopen("sales.txt", "a");
		printf("\nEnter your Name:");
		gets(name);

		while (count < 3)
		{
			printf("\nEnter your Mobile number:");
			gets(mob);
			if (strlen(mob) < 10 || strlen(mob) > 10)
			{
				printf("Invalid number\n");
			}
			else
			{
				break;
			}
			count++;
			if (count == 3)
			{
				main();
			}
		}

		_flushall();
		printf("\nEnter your Address:");
		gets(add);
		fprintf(fptr1, "\n\nName:%s ", name);
		fprintf(fptr1, "\nMobile no:%s", mob);
		fprintf(fptr1, "\nAddress:%s", add);
		printf("\n\t\t 1. Place Your Order\n");
		printf("\t\t 2. Display Final Bill\n");
		printf("\t\t 3. Back To Main Menu\n");
		printf("\t\t\nENTER YOUR CHOICE--> ");
		scanf("%d", &ent);
		switch (ent)
		{
		case 1:
		{
			print_line();
			printf("\t\t\t\t\t\tMENU\n");
			print_line();
			file_menucustomer();
			printf("\nEnter Your Choice:");
			scanf("%d", &m1);

			sum += arr[m1 - 1];

			printf("Your total bill of ordered items is:%d", sum);
			if (m1 <= 15)
			{
				printf("\nYour order is placed..\n");
				printf("\nWould you like to order something else (Y/N): ");
			}
			else if (m1 >= 15)
			{
				printf("\nInvalid input\n");
			}
			scanf("%s", &m1_1);
			while (m1_1 == 'y' || m1_1 == 'Y')
			{
				printf("\nEnter your choice:");
				scanf("%d", &m1);

				sum += arr[m1 - 1];

				if (m1 <= 15)
				{
					printf("\nWould you like to order something else (Y/N): ");
					scanf("%s", &m1_1);
				}
				else if (m1 >= 15)
				{
					printf("\nInvalid input\nTry entering correct choice");
				}
			}
			printf("\nYour order is placed..\n");
			printf("Your total bill of ordered items is: %d\n", sum);
			fprintf(fptr1, "\nBill:%d", sum);
			fclose(fptr1);
			printf("\nEnter 5 to return to the main menu:");
			scanf("%d", &ent);
			if (ent == 5)
			{
				main();
			}

			break;
		}
		break;

		case 2:
		{
			printf("\n\nName:%s ", name);
			printf("\nMobile no:%s", mob);
			printf("\nAddress:%s", add);
			printf("\nBill:%d\n", sum);

			break;
		}
		case 3:
		{
			main();
			break;
		}
		break;
		}
	case 3:
	{
		print_line();
		printf("\t\t\t\t\t\tThankyou\n");
		print_line();
		break;
	}
		return 0;
	}
	}
}

/*
Project by:
			Om Dave
			Vats Patel
			Khwahish Patel
*/
