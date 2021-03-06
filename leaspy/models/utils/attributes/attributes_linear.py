from .attributes_abstract import AttributesAbstract


# TODO 2 : Add some individual attributes -> Optimization on the w_i = A * s_i
class AttributesLinear(AttributesAbstract):
    """
    Attributes
    ----------
    dimension: `int`
    source_dimension: `int`
    betas: `torch.Tensor` (default None)
    mixing_matrix: `torch.Tensor` (default None)
        Matrix A such that w_i = A * s_i
    orthonormal_basis: `torch.Tensor` (default None)
    positions: `torch.Tensor` (default None)
    velocities: `torch.Tensor` (default None)
    name: `str` (default 'univariate')
        Name of the associated leaspy model. Used by ``update`` method.
    update_possibilities: `tuple` [`str`] (default ('all', 'g', 'v0', 'betas') )
        Contains the available parameters to update. Different models have different parameters.

    Methods
    -------
    get_attributes()
        Returns the following attributes: ``g``, ``deltas`` & ``mixing_matrix``.
    update(names_of_changed_values, values)
        Update model group average parameter(s).
    """

    def __init__(self, dimension, source_dimension):
        """
        Instantiate a AttributesLinear class object.

        Parameters
        ----------
        dimension: `int`
        source_dimension: `int`
        """
        super().__init__(dimension, source_dimension)
        self.name = 'linear'
        if (type(dimension) != int) & (type(source_dimension) != int):
            raise ValueError("For AttributesLinear you must provide integer io for the parameters"
                             " `dimension` and `source_dimension`!")

    def _compute_positions(self, values):
        """
        Update the attribute ``positions``.

        Parameters
        ----------
        values: `dict` [`str`, `torch.Tensor`]
        """
        self.positions = values['g'].clone()

    def _compute_velocities(self, values):
        """
        Update the attribute ``velocities``.

        Parameters
        ----------
        values: `dict` [`str`, `torch.Tensor`]
        """
        self.velocities = values['v0'].clone()

    def _compute_betas(self, values):
        """
        Update the attribute ``betas``.

        Parameters
        ----------
        values: `dict` [`str`, `torch.Tensor`]
        """
        self.betas = values['betas'].clone()

    def _compute_orthonormal_basis(self):
        """
        Compute the attribute ``orthonormal_basis`` which is a basis orthogonal to velocities v0 for the inner product
        implied by the metric. It is equivalent to be a base orthogonal to v0 / (p0^2 (1-p0)^2) for the euclidean norm.
        """
        dgamma_t0 = self.velocities
        self._compute_Q(dgamma_t0)
