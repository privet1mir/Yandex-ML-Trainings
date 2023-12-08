import numpy as np

class LaplaceDistribution:    
    @staticmethod
    def mean_abs_deviation_from_median(x: np.ndarray):
        '''
        Args:
        - x: A numpy array of shape (n_objects, n_features) containing the data
          consisting of num_train samples each of dimension D.
        '''
        ####
        # Do not change the class outside of this block
        if x[0].shape: 
            median = np.zeros(x.shape[1])
            deviations = np.zeros(x.shape[1])
            for j in range(x.shape[1]): 
                nums = x[:, j]
                nums.sort()
                if len(nums)%2 == 0: 
                    median[j] = (nums[len(nums)//2 - 1] + nums[len(nums)//2])/2
                else: 
                    median[j] = nums[len(nums)//2]
                deviations[j] = sum([abs(num - median[j]) for num in nums])/len(nums)
        else: 
            nums = x
            nums.sort()
            if len(nums)%2 == 0: 
                median = (nums[len(nums)//2 - 1] + nums[len(nums)//2])/2
            else: 
                median = nums[len(nums)//2]
            deviations = sum([abs(num - median) for num in nums])/len(nums)
        
        return median, deviations
        ####

    def __init__(self, features):
        '''
        Args:
            feature: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        # Do not change the class outside of this block
        self.loc =  self.mean_abs_deviation_from_median(features)[0]
        self.scale = self.mean_abs_deviation_from_median(features)[1]
        ####


    def logpdf(self, values):
        '''
        Returns logarithm of probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        # Do not change the class outside of this block
        log_pdf = values.copy()
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                log_pdf[i][j] = np.log(1/(2 * self.scale[j])) - (abs(values[i][j] - self.loc[j]) / self.scale[j])

        return log_pdf
        ####
        
    
    def pdf(self, values):
        '''
        Returns probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        return np.exp(self.logpdf(values))
