"""
Cluster class to Implement (inPython) the K-means clustering algorithm [Jin and Han, 2010]
with a Euclidean distance metric.

@Author Lunga Tsewu
@Date  21 May 2021
"""
from numpy import array
from numpy import mean
from scipy.spatial import distance
import numpy as np

"""
 * Static function to write the output(clusters) on the Cluster.txt file
 *Open file and write outputs that store in a list called output_list 
 *If file can't be open and written on it, message Can not open and write in a file on the IDE/terminal output
 *finally Done will be printed on IDE window output or terminal window 
"""
def output_into_file(output_list):

    try:
        file = open("Cluster.txt", "a")
        file.write("************************************ Cluster implements k-mean "
                   "Algorithm ***************************** "
                   "\n".upper())
        for i in output_list:
            file.write(i)

    except:
        print("Can not open and write in a file.")
    finally:
        print("Done")
        file.close()


class Cluster:
    """
    * Defaulted constructor
    * Initialise the data
    """

    def __init__(self):
        self.User_Data = array([[2, 10], [2, 5], [8, 4], [5, 8], [7, 5], [6, 4], [1, 2], [4, 9]])
        self.Mean_column = mean(self.User_Data.T, axis=1)
        self.cluster_one = array([[2, 10], [2, 5], [8, 4]])
        self.centriod_cluster_1 = array([2, 10])
        self.cluster_two = array([[5, 8], [7, 5], [6, 4]])
        self.centriod_cluster_2 = array([5, 8])
        self.cluster_three = array([[1, 2], [4, 9]])
        self.centriod_cluster_3 = array([1, 2])

    # region PCA
    """
     Get mean of cluster
     @:return mean which is a centriod 
    """
    def get_mean_column(self):
        return self.Mean_column

    """
    set mean of the cluster
    @:param cluster given cluster to calculate its mean(centriod)
    """
    def set_mean_column(self, cluster):
        self.Mean_column = mean(cluster.T, axis=1)

    # region GET & SET

    """
     Get cluster one
     @:return cluster co-ordinates
    """
    def get_cluster_one(self):
        return self.cluster_one
    """
     set cluster one
     @:param cluster given cluster
    """
    def set_cluster_one(self, cluster):
        self.cluster_one = cluster

    """
         Get cluster two
         @:return cluster co-ordinates cluster two
    """
    def get_cluster_two(self):
        return self.cluster_two

    """
        set cluster two
        @:param cluster given cluster
    """
    def set_cluster_two(self, cluster):
        self.cluster_two = cluster

    """
      Get cluster three
      @:return cluster co-ordinates cluster three
    """
    def get_cluster_three(self):
        return self.cluster_three

    """
      set cluster three
      @:param cluster given cluster
    """
    def set_cluster_three(self, cluster):
        self.cluster_three = cluster

    """
        Get cluster one
        @:return centre co-ordinates cluster one
    """
    def get_centre_cluster_one(self):
        return self.centriod_cluster_1

    def set_centre_cluster_one(self, centriod):
        self.centriod_cluster_1 = centriod

    """
      Get cluster two
      @:return centre co-ordinates cluster two
    """
    def get_centre_cluster_two(self):
        return self.centriod_cluster_2
    """
    set centre two
    @:param centriod given centre
    """
    def set_centre_cluster_two(self, centriod):
        self.centriod_cluster_2 = centriod

    """
      Get cluster three
      @:return centre co-ordinates cluster three
    """
    def get_centre_cluster_three(self):
        return self.centriod_cluster_3

    """
        set centre three
        @:param centriod given centre
    """
    def set_centre_cluster_three(self, centriod):
        self.centriod_cluster_3 = centriod

    # endregion

    # region ITERATION PROCESS
    """
    * Function to generate new centre for given data and cluster number 
    * It generates new centre of the cluster if the centriod given cluster is 
     not equals to  current cluster
    * Centriod of the given cluster is mean  function of numpy package 
    @:param Cluster is the given cluster 
    @:param i is the number of the cluster
    @:return True if the current cluster centriod is equals given cluster centriod else
    @:return False
    """
    def generate_new_centre(self, cluster, i):
        centriod = mean(cluster.T, axis=1)  # centriod of the given cluster
        x1 = centriod.item(0)
        y1 = centriod.item(1)
        if i == 1:
            x2 = self.get_centre_cluster_one().item(0)
            y2 = self.get_centre_cluster_one().item(1)
            if x1 != x2 and y1 != y2:
                self.set_centre_cluster_one(centriod)
                return False
            else:
                return True
        elif i == 2:
            x2 = self.get_centre_cluster_two().item(0)
            y2 = self.get_centre_cluster_two().item(1)
            if x1 != x2 and y1 != y2:
                self.set_centre_cluster_two(centriod)
                return False
            else:
                return True
        else:
            x2 = self.get_centre_cluster_three().item(0)
            y2 = self.get_centre_cluster_three().item(1)
            if x1 != x2 and y1 != y2:
                self.set_centre_cluster_three(centriod)
                return False
            else:
                return True



    """
       * Function to generate new clusters for each generated new centriod * To allocated each data point centriod , 
         calculated distances between data point and new centriod
       * Centriod with lesser distance from data point mean 
         that data point belong cluster of that centriod * It is stored in the centriod data for example if 
         geomatric_distance1 < geomatric_distance2 and geomatric_distance1 < geomatric_distance3  , centriod will stored
         at cluster 1 using set function of centriod 1
       * Use range of 8 because there are 8 data points
   """
    def generate_new_clusters(self):

        # set clusters to nothing
        self.set_cluster_one(np.empty((0, 2), int))
        self.set_cluster_two(np.empty((0, 2), int))
        self.set_cluster_three(np.empty((0, 2), int))

        # create new clusters
        for i in range(8):
            component = array(self.User_Data[i])
            cluster_x1 = component.item(0)
            cluster_y1 = component.item(1)

            cluster1_x2 = self.get_centre_cluster_one().item(0)
            cluster1_y2 = self.get_centre_cluster_one().item(1)
            geomatric_distance1 = distance.euclidean((cluster_x1, cluster_y1), (cluster1_x2, cluster1_y2))

            cluster2_x2 = self.get_centre_cluster_two().item(0)
            cluster2_y2 = self.get_centre_cluster_two().item(1)
            geomatric_distance2 = distance.euclidean((cluster_x1, cluster_y1), (cluster2_x2, cluster2_y2))

            cluster3_x2 = self.get_centre_cluster_three().item(0)
            cluster3_y2 = self.get_centre_cluster_three().item(1)
            geomatric_distance3 = distance.euclidean((cluster_x1, cluster_y1), (cluster3_x2, cluster3_y2))

            if geomatric_distance1 < geomatric_distance2 and geomatric_distance1 < geomatric_distance3:
                cluster = np.append(self.get_cluster_one(), np.array([component]), axis=0)
                self.set_cluster_one(cluster)

            elif geomatric_distance2 < geomatric_distance1 and geomatric_distance2 < geomatric_distance3:
                cluster = np.append(self.get_cluster_two(), np.array([component]), axis=0)
                self.set_cluster_two(cluster)

            elif geomatric_distance3 < geomatric_distance1 and geomatric_distance3 < geomatric_distance2:
                cluster = np.append(self.get_cluster_three(), np.array([component]), axis=0)
                self.set_cluster_three(cluster)

    """        
    * Check if all clusters are converging 
    *Number iteration initiated at 1 because , first iteration is where choose random centre and clusters
    * number of iteration is incremented when cluster and centriod are generated
    * if converges is True then
    *: store data and break 
    * Else generate new centroids for 3 cluster and generate new clusters 1,2 & 3
    """

    def convergence(self):
        number_iterations = 1
        out_putList = [
            "\n\n******** Centres after iteration " + str(number_iterations) + " ********\n" + self.output_format()]
        converges1 = self.generate_new_centre(self.get_cluster_one(), 1)
        converges2 = self.generate_new_centre(self.get_cluster_two(), 2)
        converges3 = self.generate_new_centre(self.get_cluster_three(), 3)
        while True:
            number_iterations += 1
            out_putList.append("\n\n******** Centres after iteration " + str(number_iterations) + " ********\n")
            if converges1 is True and converges2 is True and converges3 is True:
                out_putList.append(self.output_format() + "\n")
                out_putList.append(("Centriod Converges after " + str(number_iterations) + " iterations").upper())
                output_into_file(out_putList)
                break

            else:

                converges1 = self.generate_new_centre(self.get_cluster_one(), 1)
                converges2 = self.generate_new_centre(self.get_cluster_two(), 2)
                converges3 = self.generate_new_centre(self.get_cluster_three(), 3)
                self.generate_new_clusters()
                out_putList.append(self.output_format())



    """
       *function to format the outputs
       *Outputs will be cluster  number , centriod of the cluster, co-ordinates of the cluster
       * Use concatenation to format output
       @:return concatenated output string 
    """
    def output_format(self):

        coordnates1 = ""
        for i in self.get_cluster_one():
            coordnates1 += " " + str(i)

        coordnates2 = ""
        for i in self.get_cluster_two():
            coordnates2 += " " + str(i)

        coordnates3 = ""
        for i in self.get_cluster_three():
            coordnates3 += " " + str(i)
        myString = "Cluster 1 data:".upper() + "\n> Centriod :" + str(self.get_centre_cluster_one()) + "\n>" \
                                                                                                       " Cluster co-ordinates:\n" + coordnates1 + \
                   "\n\nCluster 2 data:".upper() + "\n> Centriod :" + str(self.get_centre_cluster_two()) + "\n>" \
                                                                                                           " Cluster co-ordinates:\n" + coordnates2 + \
                   "\n\nCluster 3 data".upper() + ":\n> Centriod :" + str(self.get_centre_cluster_three()) + "\n>" \
                                                                                                             " Cluster co-ordinates:\n" + coordnates3
        return myString

    # endregion


"""
main function to execute the class cluster
"""
def main():
    cluster = Cluster()
    cluster.convergence()
main()
