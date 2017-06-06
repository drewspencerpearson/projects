import sqlite3 as sql
import csv
import pandas as pd
import numpy as np


def prob1():
    """
    Create the following SQL tables with the following columns:
        -- MajorInfo: MajorID (int), MajorName (string)
        -- CourseInfo CourseID (int), CourseName (string)
    --------------------------------------------------------------
    Do not return anything.  Just create the designated tables.
    """
    db = sql.connect("sql1")
    cur = db.cursor()
    cur.execute("DROP TABLE IF EXISTS MajorInfo")
    cur.execute("DROP TABLE IF EXISTS CourseInfo")
    cur.execute('CREATE TABLE MajorInfo (MajorID INT NOT NULL, MajorName STRING);')
    cur.execute('CREATE TABLE CourseInfo (CourseID INT NOT NULL, CourseName STRING);')
    cur.execute("PRAGMA table_info('MajorInfo')")
    db.commit()
    db.close()

def prob2():
	"""
	Create the following SQL table with the following columns:
	-- ICD: ID_Number (int), Gender (string), Age (int) ICD_Code (string)
	--------------------------------------------------------------
	Do not return anything.  Just create the designated table.
	"""
	db = sql.connect("sql2")
	cur = db.cursor()
	cur.execute("DROP TABLE IF EXISTS ICD")
	cur.execute('CREATE TABLE ICD (ID_Number INT NOT NULL, Gender CHAR, Age Int, ICD_Code STRING);')
	with open('icd9.csv', 'rb') as csvfile:
		rows = [row for row in csv.reader(csvfile, delimiter=',')]
	cur.executemany("INSERT INTO ICD VALUES(?, ?, ?, ?);",rows)
	db.commit()
	db.close()

def prob3():
	"""
	Create the following SQL tables with the following columns:
	-- StudentInformation: StudentID (int), Name (string), MajorCode (int)
	-- StudentGrades: StudentID (int), ClassID (int), Grade (int)

	Populate these tables, as well as the tables from Problem 1, with
	the necesary information.  Also, use the column names for
	MajorInfo and CourseInfo given in Problem 1, NOT the column
	names given in Problem 3.
	------------------------------------------------------------------------
	Do not return anything.  Just create the designated tables.
	"""
	db = sql.connect("sql1")
	cur = db.cursor()
	cur.execute("DROP TABLE IF EXISTS StudentInformation")
	cur.execute('CREATE TABLE StudentInformation (StudentID INT NOT NULL, Name STRING, MajorCode INT);')
	cur.execute("DROP TABLE IF EXISTS StudentGrades")
	cur.execute('CREATE TABLE StudentGrades (StudentID INT NOT NULL, ClassID INT, Grade text);')
	cur.execute("DROP TABLE IF EXISTS MajorInfo")
	cur.execute("DROP TABLE IF EXISTS CourseInfo")
	cur.execute('CREATE TABLE MajorInfo (MajorID INT NOT NULL, MajorName STRING);')
	cur.execute('CREATE TABLE CourseInfo (CourseID INT NOT NULL, CourseName STRING);')

	with open('major_info.csv', 'rb') as csvfile:
		rows = [row for row in csv.reader(csvfile, delimiter=',')]
	cur.executemany("INSERT INTO MajorInfo VALUES(?, ?);",rows)
	with open('student_info.csv', 'rb') as csvfile:
		rows1 = [row for row in csv.reader(csvfile, delimiter=',')]
	cur.executemany("INSERT INTO StudentInformation VALUES(?, ?, ?);",rows1)
	with open('student_grades.csv', 'rb') as csvfile:
		rows2 = [row for row in csv.reader(csvfile, delimiter=',')]
	cur.executemany("INSERT INTO StudentGrades VALUES(?, ?, ?);",rows2)
	with open('course_info.csv', 'rb') as csvfile:
		rows3 = [row for row in csv.reader(csvfile, delimiter=',')]
	cur.executemany("INSERT INTO CourseInfo VALUES(?, ?);",rows3)
	db.commit()
	db.close()



    

def prob4():
	"""
	Find the number of men and women, respectively, between ages 25 and 35
	(inclusive).
	You may assume that your "sql1" and "sql2" databases have already been
	created.
	------------------------------------------------------------------------
	Returns:
	(n_men, n_women): A tuple containing number of men and number of women
	(in that order)
	"""
	db = sql.connect("sql2")
	cur = db.cursor()
	#men_count = cur.execute("Select count('Gender') from ICD where Gender = 'M' and Age >= 25 and Age <= 35;").fetchall()
	men = len(cur.execute("Select Gender from ICD where Gender = 'M' and Age >= 25 and Age <= 35;").fetchall())
	#women_count = cur.execute("Select count('Gender') from ICD where Gender = 'F' and Age >= 25 and Age <= 35;").fetchall()
	women = len(cur.execute("Select Gender from ICD where Gender = 'F' and Age >= 25 and Age <= 35;").fetchall())
	
	db.close()
	return  (men,women)

def useful_test_function(db, query):
    """
    Print out the results of a query in a nice format using pandas
    ------------------------------------------------------------------------
    Inputs:
        db: A sqlite3 database connection
        query: A string containing the SQL query you want to execute
    """
    print pd.read_sql_query(query, db)
    



if __name__ == '__main__':
	#prob1()
	#prob2()
	#prob3()
	print prob4()
	#useful_test_function("sql1", "select * from CourseInfo;")
	#useful_test_function(sql.connect("sql1"), 'SELECT * from CourseInfo')

