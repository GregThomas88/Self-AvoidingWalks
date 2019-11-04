'''
Authors: Greg Thomas and Caleb Yost
Assignment: Final Project
Project: #4
Date: 12/14/18
Description: This project calculates the number of self-avoiding walks that exist per k distance.
There are calculations for walks that maintain memory of their previous moves using DFAs.
We have implemented DFAs that maintain memory of 4, 6, and 8
'''

import numpy as np 
import math
import matplotlib.pyplot as plt

#4-memory walk
def matrix4(linenum):
    ss4 = np.matrix([1,0,0,0,0], dtype = object)

    #################0#1#2#3#4
    A4 = np.matrix([[0,4,0,0,0], #0
                    [0,1,2,0,1], #1
                    [0,1,1,1,1], #2
                    [0,1,1,0,2], #3
                    [0,0,0,0,4]], dtype = object) #4

    acs4 = np.matrix([[1],[1],[1],[1],[0]], dtype = object)

    #Calculations are handled through this
    result4 = np.linalg.matrix_power(A4,(linenum))
    temp4 = np.dot(ss4,result4)
    answer4 = np.dot(temp4,acs4)
    answer4 = str(answer4).strip("[]")
    answer4 = int(answer4)
    
    return answer4

#6-memory walk
def matrix6(linenum):
    ss6 = np.matrix([1,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype = object)

    #################0#1#2#3#4#5#6#7#8#910111213###
    A6 = np.matrix([[0,4,0,0,0,0,0,0,0,0,0,0,0,0], #0
                    [0,0,1,2,0,0,0,0,0,0,0,0,0,1], #1
                    [0,0,1,0,0,0,2,0,0,0,0,0,0,1], #2
                    [0,0,0,1,1,1,0,0,0,0,0,0,0,1], #3
                    [0,0,0,1,0,0,0,1,1,0,0,0,0,1], #4
                    [0,0,0,1,0,0,0,1,0,0,0,0,0,2], #5
                    [0,0,0,1,1,0,0,0,0,0,0,1,0,1], #6
                    [0,0,1,0,0,0,1,0,0,0,0,0,1,1], #7
                    [0,0,0,1,1,0,0,0,0,1,0,0,0,1], #8
                    [0,0,0,1,0,0,0,0,0,0,0,0,0,3], #9
                    [0,0,1,0,0,0,0,1,0,0,0,0,0,2], #10
                    [0,0,0,1,0,0,0,0,0,0,1,0,0,2], #11
                    [0,0,0,1,1,0,0,0,0,0,0,0,0,2], #12
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,4]], dtype = object) #13

    acs6 = np.matrix([[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[0]], dtype = object)

    #Calculations are handled through this
    result6 = np.linalg.matrix_power(A6,(linenum))
    temp6 = np.dot(ss6,result6)
    answer6 = np.dot(temp6,acs6)
    answer6 = str(answer6).strip("[]")
    answer6 = int(answer6)
    
    return answer6

#8-memory walk
def matrix8(linenum):

    ss8 = np.matrix([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype = object)

    #################0#1#2#3#4#5#6#7#8#9#1011121314151617181920212223
    A8 = np.matrix([[0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#0
                    [0,0,1,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],#1
                    [0,0,1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],#2
                    [0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1],#3
                    [0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],#4
                    [0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],#5
                    [0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],#6
                    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,2],#7
                    [0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1],#8
                    [0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,2],#9
                    [0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1],#10
                    [0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,2],#11
                    [0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],#12
                    [0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],#13
                    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1],#14
                    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1],#15
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,3],#16
                    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,2,0,0,0,0,1],#17
                    [0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],#18
                    [0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1],#19
                    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1],#20
                    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2],#21
                    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],#22
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4]], dtype = object) #23

    acs8 = np.matrix([[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[0]], dtype = object)

    #Calculations are handled through this
    result8 = np.linalg.matrix_power(A8,(linenum))
    temp8 = np.dot(ss8,result8)
    answer8 = np.dot(temp8,acs8)
    answer8 = str(answer8).strip("[]")
    answer8 = int(answer8)
    
    return answer8

#An equation that attempts to get close the number
def equation(linenum):
    x = (math.sqrt(182 + ((26*math.sqrt(30261))))/26)
    y = (11/32)
    answerEq = round((pow(x, linenum) * pow(linenum, y)))

    return answerEq

#Calculates the percent error
def perror(result, correct):
    pe = (((abs(result - correct))/ correct) * 100)

    return pe
    
def main():
    #These are the actual calculate values of a self-avoiding walk, the array is written this way for readability
    actualVal = [1,
                 4,
                 12,
                 36,
                 100,
                 284,
                 780,
                 2172,
                 5916,
                 16268,
                 44100,
                 120292,
                 324932,
                 881500,
                 2374444,
                 6416596,
                 17245332,
                 46466676,
                 124658732,
                 335116620,
                 897697164,
                 2408806028,
                 6444560484,
                 17266613812,
                 46146397316,
                 123481354908,
                 329712786220,
                 881317491628,
                 2351378582244,
                 6279396229332,
                 16741957935348,
                 44673816630956,
                 119034997913020,
                 317406598267076,
                 845279074648708,
                 2252534077759844,
                 5995740499124412,
                 15968852281708724,
                 42486750758210044,
                 113101676587853932]

    #UI
    print("---CS454 Final Project: Self Avoiding Walks---")
    print("Enter a number between 0 and 39, in order to compare SAW methods")
    linenum = -1 
    while (0 > linenum or linenum > 39): 
        linenum = int(input(": " ))
        if((0 > linenum) or (linenum > 39)):
            print("Error: number is not between 0 and 39")
    print("Data for the number", linenum)
    print("The number of walks for is", pow(4, linenum))
    print("The correct number self-avoiding walks is", actualVal[linenum])
    #of walks prints statements
    res4, res6, res8, reseq = matrix4(linenum), matrix6(linenum), matrix8(linenum), equation(linenum)
    print("The number of self-avoiding walks using a DFA with memory of 4:",str(res4))
    print("The number of self-avoiding walks using a DFA with memory of 6:",str(res6))
    print("The number of self-avoiding walks using a DFA with memory of 8:",str(res8))
    print("Using the equation that was found and edited:",reseq)
    #Percent Error print statements
    pe4, pe6, pe8, peeq = perror(res4, actualVal[linenum]), perror(res6, actualVal[linenum]),  perror(res8, actualVal[linenum]),  perror(reseq, actualVal[linenum])
    print("The percent error of self-avoiding walks using a DFA with memory of 4:",str(pe4) + '%')
    print("The percent error of self-avoiding walks using a DFA with memory of 6:",str(pe6) + '%')
    print("The percent error of self-avoiding walks using a DFA with memory of 8: ",str(pe8) + '%')
    print("Percent error using the equation that was found and edited:",str(peeq) + '%\n')
    response = ""
    
main()
          
