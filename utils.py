import torch

def classification_info(total, output_class, Y_class, correct):
            """
            Compares prediction to ground truth, and keeps tally of how many are correct
            """
            print(output_class[0])
            # print the index of the max value of output_class[0]
            output_class0 = torch.argmax(output_class[0])
            print('predicted_class' , output_class0)
            Y_class0 = torch.argmax(Y_class[0])
            print('True class', Y_class0)
            if Y_class0 == output_class0:
                correct += 1
            print('total', total)
            print('correct: ', correct)
            print('wrong: ', total - correct)
            print('average correct: ', correct/total)
            average_correct = correct/total

            return correct, average_correct

def regression_info(output_reg, Y_reg, criterion_reg, animal):
        """
        Compares the four coordinates to the ground truth, to keep a better overwiew
        of how correct the model is
        """

        if animal == True:
            loss = criterion_reg(output_reg, Y_reg)
        else:
            loss = [] # Think this is nice to really make sure we don't accidentally train on this 

        print('prediction bounding box:', output_reg)
        print('True bounding box:' ,Y_reg)
        print('Regression loss:', loss)

        info = 0

        return info