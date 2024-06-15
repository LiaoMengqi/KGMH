import torch


class GGEcdDcdBase(torch.nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 num_entity,
                 num_relation,
                 input_dim,
                 output_dim,
                 hidden_dims=None):
        super(GGEcdDcdBase, self, ).__init__()

        if hidden_dims is None:
            hidden_dims = []
        dims = [input_dim] + hidden_dims + [output_dim]

        self.decoder = decoder
        self.encoder = encoder
        self.rela_embed = None
        self.node_embed = None

        # encoder
        if encoder == 'gcn':
            from base_models.gcn_base import GCNBase as Encoder
            self.Encoder = Encoder(input_dim, output_dim, hidden_dims)
            self.node_embed = torch.nn.Embedding(num_entity, input_dim)

        elif encoder == 'rgcn':
            from base_models.rgcn_base import RGCNBase as Encoder
            self.Encoder = Encoder(dims, num_relation, num_entity)

        elif encoder == 'gat':
            from base_models.gat_base import GATBase as Encoder
            self.Encoder = Encoder(input_dim, output_dim, hidden_dims=hidden_dims)
            self.node_embed = torch.nn.Embedding(num_entity, input_dim)

        # decoder

        if decoder == 'distmult':
            from base_models.distmult_base import DistMultDecoder as Decoder
            self.Decoder = Decoder(num_relation, output_dim)
        elif decoder == 'transe':
            self.rela_embed = torch.nn.Embedding(num_relation, embedding_dim=output_dim)

    def forward(self, edges):
        if self.encoder == 'gcn':
            h = self.Encoder(self.node_embed.weight, edges)
            return h

        elif self.encoder == 'rgcn':
            h = self.Encoder(edges)
            return h

        elif self.encoder == 'gat':
            h = self.Encoder(self.node_embed.weight, edges)
            return h

        return
