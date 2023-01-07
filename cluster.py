import pandas as pd
import numpy as np

class distance_matrix_cluster:
    """
    Clustering of data in a matrix of distance values (like cosine distances).
    The clustering will performed based on the parameters and will create clusters with numeric cluster title, the cluster 0 will cover the elements with no cluster detected (this could be the fuzzy data not fitting to other data points). The parameters will allow to steer the cluster detection in terms of grade for similarity for clustered values and fuzzy nearby values. In addition an optimization of clusters can be selected to get clusters based on the similarity based on the data itself.

    ...

    Attributes
    ----------
    top_n : integer (top_n=5)
        number of points considered for the detaction of the cluster border. low value leads to sharper clustering.
    similar_delta : float (similar_delta=0.2)
        Value to consider 2 points as similar, the maximum distance between 2 similas points.
    optimizer : string (optimizer='min-delta')
        'none'=no optimization the maximum distance wil be taken from the similar_delta
        'min-delta'=calculate the cluster from 1 downwards with the step_size parameter and detect the clustering, with the minimum of average distance in the cluster shapes.
    step_size : (step_size=0.05)
        step_size for the optimization
    fuzzy_cluster : (fuzzy_cluster=False)
        When set to true, the clusters with no clusterdetection in the first clustering will be set to the closest cluster, if the distance is below the fuzzy_cluster_similarity. 
    fuzzy_cluster_similarity : (fuzzy_cluster_similarity=0.5)
        Maximum similarity of a closest cluster point for fuzzy data clustering.

    Methods
    -------
    fit(data)
        fit the given data (matrix for values of n*n features). preferably distance matrix.
    trasform(data)
        transform the data into a list of clusters per feature shape of n
    fit_transform(data)
        fit and transformin one step
    get_parameter(self)
        return a dict with all current parameters, including the result of the calculation and optimization values.
        in a format 
            {
                "data" : ...,  # matrix
                "min_delta": ..., # minimum similarity
                "optimizer": ..., # used optimizer
                "optimum": ..., # calculated optimum
                "max_cluster": ..., # maximum cluster distances
                "min_cluster": ..., # minimum cluster distances
                "fuzzy_distances": ..., # minimum distances of points to the next cluster
                "result_detail": ..., # detailed results as pandas dataframe
                "step_size": ..., # step_size
                "delta_steps": ..., # value list for optimization
                "result_values": ..., # clustering result
                "fuzzy_cluster": ..., # fuzzy cluster allocation
                "fuzzy_cluster_min": ..., # min value for fuzzy cluster allocation

            }


    """
    
    def __init__(self, top_n=5, similar_delta=0.2, optimizer="min-delta", step_size=0.05, fuzzy_cluster=False, fuzzy_cluster_similarity=0.5):
        # all initial settings for the ml class
        self.top_n = top_n
        self.min_delta = similar_delta
        self.step_size = step_size
        self.optimizer = optimizer
        self.optimum = [] 
        self.max_cl = []
        self.min_cl = []
        self.grp_0 = []
        self.result = None
        self.delta_values = []
        self.result_values = []
        self.fuzzy_cluster = fuzzy_cluster
        self.fuzzy_min = fuzzy_cluster_similarity

    def __pt_eval(self, value):
        pt_eval_dict = {0:"fuzzy", 1:"border", 2:"border", 3:"point"}
        return pt_eval_dict[value]

    def __cluster_matrix(self, matrix, top_n=5, min_delta=0.2):
        # get the closest point for each point
        point_1_min = matrix.min(axis=1)
        # get the top_n closest points
        indices = np.argsort(matrix)[:, :top_n]
        closest_n = matrix[:, indices][0]
        # get the median of the closest points
        closest_n_median = np.median(closest_n,axis=1)
        # get the near points
        next_points = np.argwhere(matrix < min_delta)
        # evaluation array
        eval_arr = np.column_stack((closest_n_median, point_1_min, closest_n_median < min_delta, point_1_min < min_delta))
        # create the groups
        groups = {}
        point = 0
        next_group = 1
        while point < next_points.max():
            # when the point is in the range, check the connected points and define the groups
            if (point_1_min[point] < min_delta) or (closest_n_median[point] < min_delta):
                close_points = next_points[np.where(next_points[:,0] == point)]
                close_points_list = [i[1] for i in close_points.tolist()]
                # check if a close point is already in a group
                current_group = 0
                for close_point in close_points_list:
                    if (current_group == 0) and (close_point in groups.keys()):
                        current_group = groups[close_point]
                # if no group found, create a new one
                if current_group == 0:
                    current_group = next_group
                    next_group += 1
                groups[point] = current_group
                # set groups for all close points, if not exists
                for close_point in close_points_list:
                    if not close_point in groups.keys():
                        groups[close_point] = current_group

            point += 1
        # set 0 for the rest of the points
        grouping_list = []
        related_group_dict = {}
        for i in range(len(point_1_min)):
            if not i in groups.keys():
                grouping_list.append(0)
                # check the closest point in any group
                for related_point in np.argsort(matrix[i]).tolist():
                    if related_point in groups.keys():
                        if i not in related_group_dict.keys():
                            related_group_dict[i] = groups[related_point]


            else:
                grouping_list.append(groups[i])

        related_group_list = []
        for i in range(len(point_1_min)):
            if not i in related_group_dict.keys():
                related_group_list.append(0)
            else:
                related_group_list.append(related_group_dict[i])

        # create the output for the grouping
        grouping = pd.DataFrame(eval_arr, columns=["closest_median", "closest", "median_in", "closest_in"])
        # point evaluation
        grouping["point_eval"] = grouping["closest_in"]+grouping["median_in"]*2
        grouping["point_eval_class"] = grouping["point_eval"].apply(self.__pt_eval)
        grouping["cluster"] = grouping_list 
        grouping["cluster_closest"] = related_group_list 

        return grouping
    
    def __get_matrix_slice(self, matrix, idx1, idx2):
        return matrix[:, idx1][idx2]

    def __cluster_evaluate(self, matrix, df_cluster):
        # create cluster list
        group_list = {}
        cluster_list = df_cluster["cluster"].tolist()
        for i in range(len(cluster_list)):
            if cluster_list[i] in group_list.keys():
                group_list[cluster_list[i]].append(i)
            else:
                group_list[cluster_list[i]] = [i]
        groups_max = max(group_list.keys())
        cluster_max = []
        cluster_min = []
        i = 1
        while i < groups_max:
            points_i = group_list[i]
            row = []
            row_min = []
            j = 1
            while j < groups_max:
                points_j = group_list[j]
                matrix_slice = self.__get_matrix_slice(matrix, points_i, points_j)
                row.append(matrix_slice.max())
                row_min.append(matrix_slice.min())
                j += 1
            cluster_max.append(row)
            cluster_min.append(row_min)
            i += 1
        # for the items in group 0 detect the closest distance to a cluster
        grp_0_dist = {}
        for item in group_list[0]:
            j = 1
            while j < groups_max:
                points_j = group_list[j]
                matrix_slice = self.__get_matrix_slice(matrix,[item], points_j)
                grp_0_dist[item] = matrix_slice.min()
                j += 1
        grp_0_dist_list = []
        for i in range(len(matrix)):
            if i in grp_0_dist.keys():
                grp_0_dist_list.append(grp_0_dist[i])
            else:
                grp_0_dist_list.append(0)



        max_array = np.array(cluster_max)
        np.fill_diagonal(max_array,0)

        min_array = np.array(cluster_min)
        np.fill_diagonal(min_array,0)

        grp_0_dist_arr = np.array(grp_0_dist_list)

        return max_array, min_array, grp_0_dist_arr

    def __simmilarity_cluster(self, matrix, step_size=0.05):
        list1 = []
        values_passed = []
        values = [(i+1)*step_size for i in range(int(1//step_size*0.7))]
        # print(values)
        i = 0
        maximum_passed = False
        while (i<len(values)) and (not maximum_passed):
            min_d = values[i]
            values_passed.append(min_d)
            x = self.__cluster_matrix(matrix, min_delta = min_d)
            max_cl, min_cl, grp_0 = self.__cluster_evaluate(matrix, x)
            if (len(list1) > 0) and (min_cl.mean() > min(list1)*1.05):
                maximum_passed = True
            list1.append(min_cl.mean())
            # print(min_d, min_cl.mean(), min(list1), max(x["cluster"]))
            i += 1

        optimum = values[list1.index(min(list1))]
        x = self.__cluster_matrix(matrix, min_delta = optimum)
        max_cl, min_cl, grp_0 = self.__cluster_evaluate(matrix, x)
        x["cluster_closest_dist"] = grp_0
        return x, optimum, max_cl, min_cl, grp_0, values_passed, list1
    
    def fit(self, data):
        # all fit operations, init the model
        self.data = np.copy(data)
        # check if similarity and convert to distance
        if self.data[0,0] == 1:
            self.data= 1 - self.data 
        # fill the diagonal with 100 (to get them high)
        np.fill_diagonal(self.data,100)
        # all transforming operations, using the model
        if self.optimizer == "min-delta":
            df_out, self.optimum, self.max_cl, self.min_cl, self.grp_0, self.delta_values, self.result_values = self.__simmilarity_cluster(self.data, step_size=self.step_size)
            self.output = df_out["cluster"].tolist()
            self.result = df_out
        elif self.optimizer == "no":
            df_out = self.__cluster_matrix(self.data, top_n=self.top_n, min_delta=self.min_delta)
            self.max_cl, self.min_cl, self.grp_0 = self.__cluster_evaluate(self.data, df_out)
            self.optimum = self.min_delta
            self.output = df_out["cluster"].tolist()
            self.result = df_out
        else:
            self.output = []
        
    def transform(self, data):
        transformation = []
        if self.fuzzy_cluster:
            # self.grp_0 cluster_closest
            fuzzy_distance_list = self.grp_0.copy()
            fuzzy_distance_list[fuzzy_distance_list > self.fuzzy_min] = 0
            fuzzy_distance_list[fuzzy_distance_list != 0] = 1
            additional_fuzzy_cluster = np.array(self.result["cluster_closest"].tolist())*fuzzy_distance_list
            transformation = np.array(self.output) + additional_fuzzy_cluster
        else:
            transformation = np.array(self.output)
        
        return transformation
    
    def fit_transform(self, data):
        # all fit operations, init the model
        self.fit(data)
        return self.transform(data)
    
    def get_parameter(self):
        # get the model parameter
        parameter = {
            "data" : self.data,
            "min_delta": self.min_delta,
            "optimizer": self.optimizer,
            "optimum": self.optimum,
            "max_cluster": self.max_cl,
            "min_cluster": self.min_cl,
            "fuzzy_distances": self.grp_0,
            "result_detail": self.result,
            "step_size": self.step_size,
            "delta_steps": self.delta_values,
            "result_values": self.result_values,
            "fuzzy_cluster": self.fuzzy_cluster,
            "fuzzy_cluster_min": self.fuzzy_min,

        }
        return parameter