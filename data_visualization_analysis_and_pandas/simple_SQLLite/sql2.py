import sqlite3 as sql
import csv
import pandas as pd

def prob1():
    """
    Specify relationships between columns in given sql tables.
    """
    print "One-to-one relationships:"
    # Put print statements specifying one-to-one relationships between table
    # columns. 
    print "In table 5.1 Student ID - Name\n In table 5.2 Id - Name"

    print "**************************"
    print "One-to-many relationships:"
    print " In table 5.3 StudentId - Grade"
    # Put print statements specifying one-to-many relationships between table
    # columns.

    print "***************************"
    print "Many-to-Many relationships:"
    # Put print statements specifying many-to-many relationships between table
    # columns.

def prob2():
    """
    Write a SQL query that will output how many students belong to each major,
    including students who don't have a major.

    Return: A table indicating how many students belong to each major.
    """
    #Build your tables and/or query here
    db = sql.connect("sql3")
    cur = db.cursor()
    cur.execute("DROP TABLE IF EXISTS Students")
    cur.execute('CREATE TABLE Students (StudentID INT NOT NULL, StudentName STRING, MajorCode INT, MinorCode INT);')
    cur.execute("DROP TABLE IF EXISTS Fields")
    cur.execute('CREATE TABLE Fields (FieldID INT NOT NULL, FieldName STRING);')
    
    cur.execute("DROP TABLE IF EXISTS Grades")
    cur.execute('CREATE TABLE Grades (StudentID INT NOT NULL, ClassID STRING, Grade STRING);')
    
    cur.execute("DROP TABLE IF EXISTS Classes")
    cur.execute('CREATE TABLE Classes (ClassID INT NOT NULL, ClassName STRING);')
    
    with open('students.csv', 'rb') as csvfile:
        rows = [row for row in csv.reader(csvfile, delimiter=',')]
    cur.executemany("INSERT INTO Students VALUES(?,?,?,?);",rows)
    with open('fields.csv', 'rb') as csvfile:
        rows1 = [row for row in csv.reader(csvfile, delimiter=',')]
    cur.executemany("INSERT INTO Fields VALUES(?, ?);",rows1)
    
    with open('grades.csv', 'rb') as csvfile:
        rows2 = [row for row in csv.reader(csvfile, delimiter=',')]
    cur.executemany("INSERT INTO Grades VALUES(?, ?, ?);",rows2)
    
    with open('classes.csv', 'rb') as csvfile:
        rows3 = [row for row in csv.reader(csvfile, delimiter=',')]
    cur.executemany("INSERT INTO Classes VALUES(?, ?);",rows3)

    query = "select Fields.FieldName as FieldName, count(Students.StudentID) as number_of_students from Students left join Fields on Fields.FieldID = Students.MajorCode Group by FieldName;"


    # This line will make a pretty table with the results of your query.
        ### query is a string containing your sql query
        ### db is a sql database connection
    result =  pd.read_sql_query(query, db)
    db.commit()
    db.close()
    return result


def prob3():
    """
    Select students who received two or more non-Null grades in their classes.

    Return: A table of the students' names and the grades each received.
    """
    #Build your tables and/or query here
    db = sql.connect("sql3")
    cur = db.cursor()
    cur.execute("DROP TABLE IF EXISTS Students")
    cur.execute('CREATE TABLE Students (StudentID INT NOT NULL, StudentName STRING, MajorCode INT, MinorCode INT);')
    cur.execute("DROP TABLE IF EXISTS Fields")
    cur.execute('CREATE TABLE Fields (FieldID INT NOT NULL, FieldName STRING);')
    
    cur.execute("DROP TABLE IF EXISTS Grades")
    cur.execute('CREATE TABLE Grades (StudentID INT NOT NULL, ClassID STRING, Grade STRING);')
    
    cur.execute("DROP TABLE IF EXISTS Classes")
    cur.execute('CREATE TABLE Classes (ClassID INT NOT NULL, ClassName STRING);')
    
    with open('students.csv', 'rb') as csvfile:
        rows = [row for row in csv.reader(csvfile, delimiter=',')]
    cur.executemany("INSERT INTO Students VALUES(?,?,?,?);",rows)
    with open('fields.csv', 'rb') as csvfile:
        rows1 = [row for row in csv.reader(csvfile, delimiter=',')]
    cur.executemany("INSERT INTO Fields VALUES(?, ?);",rows1)
    
    with open('grades.csv', 'rb') as csvfile:
        rows2 = [row for row in csv.reader(csvfile, delimiter=',')]
    cur.executemany("INSERT INTO Grades VALUES(?, ?, ?);",rows2)
    
    with open('classes.csv', 'rb') as csvfile:
        rows3 = [row for row in csv.reader(csvfile, delimiter=',')]
    cur.executemany("INSERT INTO Classes VALUES(?, ?);",rows3)


    query = "select Students.StudentName, count(Grades.Grade) from Students left join Grades on Students.StudentID = Grades.StudentID where Grades.Grade is not 'NULL' group by students.StudentName having count(Grades.Grade) >2 "
    # This line will make a pretty table with the results of your query.
        ### query is a string containing your sql query
        ### db is a sql database connection
    result =  pd.read_sql_query(query, db)
    db.commit()
    db.close()
    return result


def prob4():
    """
    Get the average GPA at the school using the given tables.

    Return: A float representing the average GPA, rounded to 2 decimal places.
    """
    db = sql.connect("sql3")
    cur = db.cursor()
    cur.execute("DROP TABLE IF EXISTS Students")
    cur.execute('CREATE TABLE Students (StudentID INT NOT NULL, StudentName STRING, MajorCode INT, MinorCode INT);')
    cur.execute("DROP TABLE IF EXISTS Fields")
    cur.execute('CREATE TABLE Fields (FieldID INT NOT NULL, FieldName STRING);')
    
    cur.execute("DROP TABLE IF EXISTS Grades")
    cur.execute('CREATE TABLE Grades (StudentID INT NOT NULL, ClassID STRING, Grade STRING);')
    
    cur.execute("DROP TABLE IF EXISTS Classes")
    cur.execute('CREATE TABLE Classes (ClassID INT NOT NULL, ClassName STRING);')
    
    with open('students.csv', 'rb') as csvfile:
        rows = [row for row in csv.reader(csvfile, delimiter=',')]
    cur.executemany("INSERT INTO Students VALUES(?,?,?,?);",rows)
    with open('fields.csv', 'rb') as csvfile:
        rows1 = [row for row in csv.reader(csvfile, delimiter=',')]
    cur.executemany("INSERT INTO Fields VALUES(?, ?);",rows1)
    
    with open('grades.csv', 'rb') as csvfile:
        rows2 = [row for row in csv.reader(csvfile, delimiter=',')]
    cur.executemany("INSERT INTO Grades VALUES(?, ?, ?);",rows2)
    
    with open('classes.csv', 'rb') as csvfile:
        rows3 = [row for row in csv.reader(csvfile, delimiter=',')]
    cur.executemany("INSERT INTO Classes VALUES(?, ?);",rows3)

    cur.execute("select round(avg(case Grade when 'A+' then 4 when 'A' then 4 \
                                    when 'A-'  then 4 \
                                    when 'B+' then 3\
                                   when 'B' then 3 \
                                   when 'B-' then 3 \
                                   when 'C+' then 2 \
                                   when 'C' then 2 \
                                   when 'C-' then 2 \
                                   when 'D+' then 1 \
                                   when 'D' then 1 \
                                   when 'D-' then 1 \
                                   Else 0\
                                   End), 2)\
                                   from Grades where Grade is not 'NULL';")
    


    return cur.fetchall()[0][0]
    db.commit()
    db.close()

def prob5():
    """
    Find all students whose last name begins with 'C' and their majors.

    Return: A table containing the names of the students and their majors.
    """
    #Build your tables and/or query here
    db = sql.connect("sql3")
    cur = db.cursor()
    cur.execute("DROP TABLE IF EXISTS Students")
    cur.execute('CREATE TABLE Students (StudentID INT NOT NULL, StudentName STRING, MajorCode INT, MinorCode INT);')
    cur.execute("DROP TABLE IF EXISTS Fields")
    cur.execute('CREATE TABLE Fields (FieldID INT NOT NULL, FieldName STRING);')
    
    cur.execute("DROP TABLE IF EXISTS Grades")
    cur.execute('CREATE TABLE Grades (StudentID INT NOT NULL, ClassID STRING, Grade STRING);')
    
    cur.execute("DROP TABLE IF EXISTS Classes")
    cur.execute('CREATE TABLE Classes (ClassID INT NOT NULL, ClassName STRING);')
    
    with open('students.csv', 'rb') as csvfile:
        rows = [row for row in csv.reader(csvfile, delimiter=',')]
    cur.executemany("INSERT INTO Students VALUES(?,?,?,?);",rows)
    with open('fields.csv', 'rb') as csvfile:
        rows1 = [row for row in csv.reader(csvfile, delimiter=',')]
    cur.executemany("INSERT INTO Fields VALUES(?, ?);",rows1)
    
    with open('grades.csv', 'rb') as csvfile:
        rows2 = [row for row in csv.reader(csvfile, delimiter=',')]
    cur.executemany("INSERT INTO Grades VALUES(?, ?, ?);",rows2)
    
    with open('classes.csv', 'rb') as csvfile:
        rows3 = [row for row in csv.reader(csvfile, delimiter=',')]
    cur.executemany("INSERT INTO Classes VALUES(?, ?);",rows3)



    query = "select Students.StudentName as StudentName, Fields.FieldName as FieldName from Students left join Fields on Fields.FieldID = Students.MajorCode where Students.StudentName like '% C%'"
    # This line will make a pretty table with the results of your query.
        ### query is a string containing your sql query
        ### db is a sql database connection
    
    result =  pd.read_sql_query(query, db)
    db.commit()
    db.close()
    return result 



if __name__ == '__main__':
    #print prob2()
    #print prob3()
    print prob4()
    #print prob5()
    #print prob4()





