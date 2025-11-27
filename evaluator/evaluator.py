import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


class Evaluator:
    """
    A class for evaluating a biometric system's performance.
    """

    def __init__(self, 
                 num_thresholds, 
                 genuine_scores, 
                 impostor_scores, 
                 plot_title, 
                 epsilon=1e-12):
        """
        Initialize the Evaluator object.

        Parameters:
        - num_thresholds (int): Number of thresholds to evaluate.
        - genuine_scores (array-like): Genuine scores for evaluation.
        - impostor_scores (array-like): Impostor scores for evaluation.
        - plot_title (str): Title for the evaluation plots.
        - epsilon (float): A small value to prevent division by zero.
        """
        self.num_thresholds = num_thresholds
        self.thresholds = np.linspace(0, 1, num_thresholds)
        self.genuine_scores = genuine_scores
        self.impostor_scores = impostor_scores
        self.plot_title = plot_title
        self.epsilon = epsilon

    def get_dprime(self):
        
        """
        Calculate the d' (d-prime) metric.

        Returns:
        - float: The calculated d' value.
        """
        x = np.mean(self.genuine_scores) - np.mean(self.impostor_scores) # Provides the mean difference between genuine and imposter scores
        y = np.sqrt(0.5 * (np.std(self.genuine_scores)**2 + np.std(self.impostor_scores)**2)) #Provides the square root of 1/2 * the standard deviation^2 of both the genuine and imposter scores added together
        return x / (y + self.epsilon)

    def plot_score_distribution(self):
        """
        Plot the distribution of genuine and impostor scores.
        """
        plt.figure()
        
        # Plot the histogram for genuine scores
        plt.hist(
            x=self.genuine_scores,
            color='green',
            lw=0.5,
            histtype='step',
            hatch='///',
            label='Genuine Scores'
            
            # Provide genuine scores data here
            # color: Set the color for genuine scores
            # lw: Set the line width for the histogram
            # histtype: Choose 'step' for a step histogram
            # hatch: Choose a pattern for filling the histogram bars
            # label: Provide a label for genuine scores in the legend
        )
        
        # Plot the histogram for impostor scores
        plt.hist(
            x=self.impostor_scores,
            color='red',
            lw=0.5,
            histtype='step',
            hatch='||',
            label='Imposter Scores'
            # Provide impostor scores data here
            # color: Set the color for impostor scores
            # lw: Set the line width for the histogram
            # histtype: Choose 'step' for a step histogram
            # hatch: Choose a pattern for filling the histogram bars
            # label: Provide a label for impostor scores in the legend
        )
        
        # Set the x-axis limit to ensure the histogram fits within the correct range
        plt.xlim([-0.05, 1.05])
        
        # Add grid lines for better readability
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        
        # Step 6: Add legend to the upper left corner with a specified font size
        plt.legend(
            loc='upper left',
            fontsize=10
        )
        
        # Step 7: Set x and y-axis labels with specified font size and weight
        plt.xlabel(
            # Provide the x-axis label
            'Matching Score',
            fontsize=15,
            weight=12
        )
        
        plt.ylabel(
            # Provide the y-axis label
            'Score Frequency',
            fontsize=15,
            weight=12
        )
        
        # Step 8: Remove the top and right spines for a cleaner appearance
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Step 9: Set font size for x and y-axis ticks
        plt.xticks(
            fontsize=12
        )
        
        plt.yticks(
            fontsize=12
        )
        
        # Step 10: Add a title to the plot with d-prime value and system title
        plt.title('Score Distribution Plot\nd-prime= %.2f\nSystem %s' % 
                  (self.get_dprime(), 
                   self.plot_title),
                  fontsize=15,
                  weight='bold')
        
        # Save the figure before displaying it
        plt.savefig('score_distribution_plot_(%s).png' % self.plot_title, dpi=300, bbox_inches="tight")
        
        # Display the plot after saving
        plt.show()
        
        # Close the figure to free up resources
        plt.close()

        return

    def get_EER(self, FPR, FNR):
        """
        Calculate the Equal Error Rate (EER).
    
        Parameters:
        - FPR (list or array-like): False Positive Rate values.
        - FNR (list or array-like): False Negative Rate values.
    
        Returns:
        - float: Equal Error Rate (EER).
        """
        Ldiff = np.abs(FPR[0] - FNR[0]) #Sets the lowest difference to the difference between the first values 
        threshold = 0 #The threshold for the lowest difference
        for i in range(1,self.num_thresholds): #Finds the lowest difference between FPR and FNR for all thresholds
            diff = np.abs(FPR[i]-FNR[i])
            if diff < Ldiff:
                Ldiff = diff
                threshold = i
    
        EER = round(((FPR[threshold] + FNR[threshold]) / 2.0),5) #Averages the values at the lowest threshold to get the EER
        EER_threshold = self.thresholds[threshold]
        # Add code here to compute the EER
        
        return EER, EER_threshold

    def plot_det_curve(self, FPR, FNR):
        """
        Plot the Detection Error Tradeoff (DET) curve.
        Parameters:
         - FPR (list or array-like): False Positive Rate values.
         - FNR (list or array-like): False Negative Rate values.
        """
        
        # Calculate the Equal Error Rate (EER) using the get_EER method
        EER, EER_threshold = self.get_EER(FPR, FNR)
        
        # Create a new figure for plotting
        plt.figure()
        
        # Plot the Detection Error Tradeoff Curve
        plt.plot(
            FPR,
            FNR,
            lw=1,
            color='black'
        )
        
        # Add a text annotation for the EER point on the curve
        # Plot the diagonal line representing random classification
        # Scatter plot to highlight the EER point on the curve

        plt.text(EER + 0.07, EER + 0.07, "EER", style='italic', fontsize=12,
                 bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
        plt.plot([0, 1], [0, 1], '--', lw=0.5, color='black')
        plt.scatter([EER], [EER], c="black", s=100)
        
        # Set the x and y-axis limits to ensure the plot fits within the range 
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        
        # Add grid lines for better readability
        plt.grid(
            color='grey',
            linestyle= '--',
            linewidth=0.5
        )
        
        # Remove the top and right spines for a cleaner appearance
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Set x and y-axis labels with specified font size and weight
        plt.xlabel(
            'False Pos. Rate',
            fontsize= 15,
            weight= 12
        )
        
        plt.ylabel(
            'False Neg. Rate',
            fontsize= 15,
            weight= 12
        )
        
        # Step 11: Add a title to the plot with EER value and system title
        plt.title(
            'Detection Error Tradeoff Curve \nEER = %.5f at t=%.3f\nSystem %s'%(
            EER,
            EER_threshold,
            self.plot_title),
            fontsize= 15,
            weight='bold'
        )
        
        # Step 12: Set font size for x and y-axis ticks
        plt.xticks(
            fontsize= 12
        )
        
        plt.yticks(
            fontsize= 12
        )
        
        # Step 13: Save the plot as an image file
        plt.savefig(
            'Detection_Error_Tradeoff_Curve_(%s).png' % self.plot_title, dpi=300, bbox_inches="tight"
        )
        
        # Step 14: Display the plot
        plt.show()    
        
        # Step 15: Close the plot to free up resources
        plt.close()
    
        return

    def plot_roc_curve(self, FPR, TPR):
        """
        Plot the Receiver Operating Characteristic (ROC) curve.
        Parameters:
        - FPR (list or array-like): False Positive Rate values.
        - TPR (list or array-like): True Positive Rate values.
        """
        
        # Create a new figure for plotting
        plt.figure()
        
        # Plot the Receiver Operating Characteristic Curve
        plt.plot(
            FPR,
            TPR,
            lw=1,
            color='black'
        )
        
        
        # Plot the diagonal line representing random classification
        plt.plot([0, 1], [0, 1], '--', lw=0.5, color='black')
        
        # Set the x and y-axis limits to ensure the plot fits within the range 
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        
        # Add grid lines for better readability
        plt.grid(
            color='grey',
            linestyle= '--',
            linewidth=0.5
        )
        
        # Remove the top and right spines for a cleaner appearance
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Set x and y-axis labels with specified font size and weight
        plt.xlabel(
            'False Pos. Rate',
            fontsize= 15,
            weight= 12
        )
        
        plt.ylabel(
            'True Pos. Rate',
            fontsize= 15,
            weight= 12
        )
        
        # Step 11: Add a title to the plot with system title
        plt.title(
            'Receiver Operating Characteristic Curve \nSystem %s'%(
            self.plot_title),
            fontsize= 15,
            weight='bold'
        )
        
        # Step 12: Set font size for x and y-axis ticks
        plt.xticks(
            fontsize= 12
        )
        
        plt.yticks(
            fontsize= 12
        )
        
        # Step 13: Save the plot as an image file
        plt.savefig(
            'Receiver_Operating_Characteristic_Curve_(%s).png' % self.plot_title, dpi=300, bbox_inches="tight"
        )
        
        # Step 14: Display the plot
        plt.show()    
        
        # Step 15: Close the plot to free up resources
        plt.close()
    
        # Create a new figure for the ROC curve
        # Plot the ROC curve using FPR and TPR with specified attributes
        # Set x and y axis limits, add grid, and remove top and right spines
        # Set labels for x and y axes, and add a title
        # Set font sizes for ticks, x and y labels
        # Save the plot as a PNG file and display it
        # Close the figure to free up resources
 
        return

    def compute_rates(self):
        FPR=np.array([])
        FNR=np.array([])
        TPR=np.array([])
        #step = float(1/self.num_thresholds)
        for i in self.thresholds:
            fpc = sum(1 for val in self.impostor_scores if val >= i) #Gets the count of all imposter values that are at or above the threshold
            fnc = sum(1 for val in self.genuine_scores if val < i) #Gets the count of all genuine values that are below the threshold
            tpc = sum(1 for val in self.genuine_scores if val >= i) #Gets the count of all genuine values that are at or above the threshold
        
            FPR = np.append(FPR, (fpc/len(self.impostor_scores))) #appends the FPR to the array at each threshold
            FNR = np.append(FNR, (fnc/len(self.genuine_scores))) #appends the FNR to the array at each threshold
            TPR = np.append(TPR, (tpc/len(self.genuine_scores))) #appends the TPR to the array at each threshold
        
        return FPR, FNR, TPR
        # Initialize lists for False Positive Rate (FPR), False Negative Rate (FNR), and True Positive Rate (TPR)
        # Iterate through threshold values and calculate TP, FP, TN, and FN for each threshold
        # Calculate FPR, FNR, and TPR based on the obtained values
        # Append calculated rates to their respective lists
        # Return the lists of FPR, FNR, and TPR

def evaluate(genuine_scores, impostor_scores):
    
    seed = 71
    rng = np.random.default_rng(seed)

   
    # Creating an instance of the Evaluator class
    evaluator = Evaluator(
        epsilon=1e-12,
        num_thresholds=200,
        genuine_scores=genuine_scores,
        impostor_scores=impostor_scores,
        plot_title="CAL"
    )
        
        # Generate the FPR, FNR, and TPR using 200 threshold values equally spaced
        # between -0.1 and 1.1.
    FPR, FNR, TPR = evaluator.compute_rates()
        #print(FPR)
        #print(FNR)
        #print(TPR)
    
        # Plot the score distribution. Include the d-prime value in the plot’s 
        # title. Your genuine scores should be green, and your impostor scores 
        # should be red. Set the x axis limits from -0.05 to 1.05
    evaluator.plot_score_distribution()
                
        # Plot the DET curve and include the EER in the plot’s title. 
        # Set the x and y axes limits from -0.05 to 1.05.
    evaluator.plot_det_curve(FPR, FNR)
        
        # Plot the ROC curve. Set the x and y axes limits from -0.05 to 1.05.
    evaluator.plot_roc_curve(FPR, TPR)