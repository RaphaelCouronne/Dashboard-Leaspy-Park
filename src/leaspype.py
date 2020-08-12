import numpy as np
import warnings

def get_reparametrized_ages(ages, individual_parameters, leaspy):
    r"""
    Reparametrize the real ages of the patients onto the pathological timeline

    Parameters
    ----------
    individual_parameters: Individual parameters object
        Contains the individual parameters for each patient

    ages: dict {patient_idx: [ages]}
        Contains the patient ages to reparametrized

    leaspy: Leaspy object
        Contains the model parameters

    Returns
    -------
    reparametrized_ages: dict {patient_idx: [reparametrized_ages]}
        Contains the reparametrized ages

    Raise:
    ------
    ValueError:
        If one of the index not in the individual parameters

    Examples
    --------

    >>> ages = {'idx-1': [78, 79, 81], 'idx-2': [67, 68, 74], 'idx-3': [56]}
    >>> repametrized_ages = get_reparametrized_ages(ages, individual_parameters, leaspy)
    """

    warnings.warn('get_reparametrized_ages function is deprecated. Please use the one in Leaspype')
    tau_mean = leaspy.model.parameters['tau_mean']
    indices = individual_parameters._indices
    reparametrized_ages = {}

    for idx, ages in ages.items():
        if idx not in indices:
            raise ValueError(f'The index {idx} is not in the individual parameters')

        idx_ip = individual_parameters[idx]
        alpha = np.exp(idx_ip['xi'])
        tau = idx_ip['tau']

        reparam_ages = [alpha * (age - tau ) + tau_mean for age in ages]
        reparametrized_ages[idx] = [_.numpy().tolist() for _ in reparam_ages]

    return reparametrized_ages



def append_spaceshifts_to_individual_parameters_dataframe(df_individual_parameters, leaspy):
    r"""
    Returns a new dataframe with space shift columns

    Parameters
    ----------
    df_individual_parameters: pandas.Dataframe
        Dataframe of the individual parameters. Each row corresponds to an individual. The index is the index of the patient.
    leaspy: Leaspy
        Initialize model

    Returns
    -------
    dataframe: pandas.Dataframe
        Copy of the initial dataframe with additional columns being the space shifts of the individuals.

    """
    warnings.warn('append_spaceshifts_to_individual_parameters_dataframe function is deprecated. Please use the one in Leaspype')
    df_ip = df_individual_parameters.copy()

    sources = df_ip [['sources_' + str(i) for i in range(leaspy.model.source_dimension)]].values.T
    spaceshifts = np.dot(leaspy.model.attributes.mixing_matrix, sources)

    for i, spaceshift_coord in enumerate(spaceshifts):
        df_ip['w_' + str(i)] = spaceshift_coord

    return df_ip

