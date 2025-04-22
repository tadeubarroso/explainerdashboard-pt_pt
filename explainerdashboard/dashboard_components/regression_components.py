__all__ = [
    "RegressionRandomIndexComponent",
    "RegressionPredictionSummaryComponent",
    "PredictedVsActualComponent",
    "ResidualsComponent",
    "RegressionVsColComponent",
    "RegressionModelSummaryComponent",
]

import numpy as np
import pandas as pd

import dash
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from ..dashboard_methods import *
from .. import to_html



class RegressionRandomIndexComponent(ExplainerComponent):
    _state_props = dict(index=("random-index-reg-index-", "value"))

    def __init__(
        self,
        explainer,
        title=None, # Será definido abaixo se for None
        name=None,
        subtitle="Selecione da lista ou escolha aleatoriamente",
        hide_title=False,
        hide_subtitle=False,
        hide_index=False,
        hide_pred_slider=False,
        hide_residual_slider=False,
        hide_pred_or_y=False,
        hide_abs_residuals=False,
        hide_button=False,
        index_dropdown=True,
        index=None,
        pred_slider=None,
        y_slider=None,
        residual_slider=None,
        abs_residual_slider=None,
        pred_or_y="preds",
        abs_residuals=True,
        round=2,
        description=None,
        **kwargs,
    ):
        """Componente para selecionar um índice aleatório sujeito a restrições

        Args:
            explainer (Explainer): objeto explainer construído com
                        ClassifierExplainer() ou RegressionExplainer()
            title (str, optional): Título do separador ou página. Predefinição para
                        "Selecionar Índice Aleatório".
            name (str, optional): nome único a adicionar aos elementos do Componente.
                        Se None, um uuid aleatório é gerado para garantir
                        que é único. Predefinição para None.
            subtitle (str): subtítulo
            hide_title (bool, optional): ocultar título
            hide_subtitle (bool, optional): Ocultar subtítulo. Predefinição para False.
            hide_index (bool, optional): Ocultar seletor de índice.
                        Predefinição para False.
            hide_pred_slider (bool, optional): Ocultar controlo deslizante de previsão.
                        Predefinição para False.
            hide_residual_slider (bool, optional): ocultar controlo deslizante de resíduos.
                        Predefinição para False.
            hide_pred_or_y (bool, optional): ocultar alternador de previsão ou real.
                        Predefinição para False.
            hide_abs_residuals (bool, optional): ocultar alternador de resíduos absolutos.
                        Predefinição para False.
            hide_button (bool, optional): ocultar botão. Predefinição para False.
            index_dropdown (bool, optional): Usar lista pendente para entrada de índice em vez
                de entrada de texto livre. Predefinição para True.
            index ({str, int}, optional): Índice inicial a exibir.
                        Predefinição para None.
            pred_slider ([lb, ub], optional): Valores iniciais para o controlo deslizante de valores
                        de previsão [limite inferior, limite superior]. Predefinição para None.
            y_slider ([lb, ub], optional): Valores iniciais para o controlo deslizante y
                        [limite inferior, limite superior]. Predefinição para None.
            residual_slider ([lb, ub], optional): Valores iniciais para o controlo deslizante de resíduos
                        [limite inferior, limite superior]. Predefinição para None.
            abs_residual_slider ([lb, ub], optional): Valores iniciais para o controlo deslizante
                        de resíduos absolutos [limite inferior, limite superior]
                        Predefinição para None.
            pred_or_y (str, {'preds', 'y'}, optional): Uso inicial de previsões
                        ou controlo deslizante y. Predefinição para "preds".
            abs_residuals (bool, optional): Uso inicial de resíduos ou resíduos
                        absolutos. Predefinição para True.
            round (int, optional): arredondamento usado para espaçamento do controlo deslizante. Predefinição para 2.
            description (str, optional): Dica a exibir ao passar o rato sobre
                o título do componente. Quando None, o texto predefinido é mostrado.
        """
        super().__init__(explainer, title or f"Selecionar {explainer.index_name}", name) # Traduzido o valor predefinido do título
        assert self.explainer.is_regression, (
            "explainer não é um RegressionExplainer, pelo que o RegressionRandomIndexComponent "
            "não funcionará. Tente usar o ClassifierRandomIndexComponent em vez disso."
        )

        # if self.title is None:
        #     self.title = f"Selecionar {self.explainer.index_name}" # Traduzido

        self.index_name = "random-index-reg-index-" + self.name
        self.index_selector = IndexSelector(
            explainer,
            self.index_name,
            index=index,
            index_dropdown=index_dropdown,
            **kwargs,
        )

        if self.explainer.y_missing:
            self.hide_residual_slider = True
            self.hide_pred_or_y = True
            self.hide_abs_residuals = True
            self.pred_or_y = "preds"
            self.y_slider = [0.0, 1.0]
            self.residual_slider = [0.0, 1.0]
            self.abs_residual_slider = [0.0, 1.0]

        if self.pred_slider is None:
            self.pred_slider = [
                float(self.explainer.preds.min()),
                float(self.explainer.preds.max()),
            ]

        if not self.explainer.y_missing:
            if self.y_slider is None:
                self.y_slider = [
                    float(self.explainer.y.min()),
                    float(self.explainer.y.max()),
                ]

            if self.residual_slider is None:
                self.residual_slider = [
                    float(self.explainer.residuals.min()),
                    float(self.explainer.residuals.max()),
                ]

            if self.abs_residual_slider is None:
                self.abs_residual_slider = [
                    float(self.explainer.abs_residuals.min()),
                    float(self.explainer.abs_residuals.max()),
                ]

            assert (
                len(self.pred_slider) == 2
                and self.pred_slider[0] <= self.pred_slider[1]
            ), "pred_slider deve ser uma lista de [limite_inferior, limite_superior]!"

            assert (
                len(self.y_slider) == 2 and self.y_slider[0] <= self.y_slider[1]
            ), "y_slider deve ser uma lista de [limite_inferior, limite_superior]!"

            assert (
                len(self.residual_slider) == 2
                and self.residual_slider[0] <= self.residual_slider[1]
            ), "residual_slider deve ser uma lista de [limite_inferior, limite_superior]!"

            assert (
                len(self.abs_residual_slider) == 2
                and self.abs_residual_slider[0] <= self.abs_residual_slider[1]
            ), "abs_residual_slider deve ser uma lista de [limite_inferior, limite_superior]!"

        self.y_slider = [float(y) for y in self.y_slider]
        self.pred_slider = [float(p) for p in self.pred_slider]
        self.residual_slider = [float(r) for r in self.residual_slider]
        self.abs_residual_slider = [float(a) for a in self.abs_residual_slider]

        assert self.pred_or_y in {
            "preds",
            "y",
        }, "pred_or_y deve estar em ['preds', 'y']!"

        if self.description is None:
            self.description = f"""
        Pode selecionar um {self.explainer.index_name} diretamente escolhendo-o
        na lista pendente (se começar a escrever, pode pesquisar dentro da lista),
        ou clique no botão '{self.explainer.index_name} Aleatório' para selecionar aleatoriamente
        um {self.explainer.index_name} que cumpra as restrições. Por exemplo,
        pode selecionar um {self.explainer.index_name} com um {self.explainer.target} previsto
        muito alto, ou um {self.explainer.target} observado muito baixo,
        ou um {self.explainer.index_name} cujo {self.explainer.target} previsto
        estava muito longe do {self.explainer.target} observado e, por isso, tinha um
        resíduo (absoluto) alto.
        """ # Traduzido

    def layout(self):
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title, # Já traduzido no __init__
                                        id="random-index-reg-title-" + self.name,
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle" # Já traduzido no __init__
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description, # Já traduzido no __init__
                                        target="random-index-reg-title-" + self.name,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    hide=self.hide_title,
                ),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col([self.index_selector.layout()], md=8),
                                    hide=self.hide_index,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Button(
                                                f"{self.explainer.index_name} Aleatório", # Traduzido
                                                color="primary",
                                                id="random-index-reg-button-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                f"Selecionar um {self.explainer.index_name} aleatório de acordo com as restrições", # Traduzido
                                                target="random-index-reg-button-"
                                                + self.name,
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    hide=self.hide_button,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    dbc.Label(
                                                        "Intervalo previsto:", # Traduzido
                                                        id="random-index-reg-pred-slider-label-"
                                                        + self.name,
                                                        html_for="random-index-reg-pred-slider-"
                                                        + self.name,
                                                    ),
                                                    dbc.Tooltip(
                                                        f"Selecionar apenas {self.explainer.index_name} onde o "
                                                        f"{self.explainer.target} previsto estava dentro do seguinte intervalo:", # Traduzido
                                                        target="random-index-reg-pred-slider-label-"
                                                        + self.name,
                                                    ),
                                                    dcc.RangeSlider(
                                                        id="random-index-reg-pred-slider-"
                                                        + self.name,
                                                        min=float(
                                                            self.explainer.preds.min()
                                                        ),
                                                        max=float(
                                                            self.explainer.preds.max()
                                                        ),
                                                        step=np.float_power(
                                                            10, -self.round
                                                        ),
                                                        value=[
                                                            self.pred_slider[0],
                                                            self.pred_slider[1],
                                                        ],
                                                        marks={
                                                            float(
                                                                self.explainer.preds.min()
                                                            ): str(
                                                                np.round(
                                                                    self.explainer.preds.min(),
                                                                    self.round,
                                                                )
                                                            ),
                                                            float(
                                                                self.explainer.preds.max()
                                                            ): str(
                                                                np.round(
                                                                    self.explainer.preds.max(),
                                                                    self.round,
                                                                )
                                                            ),
                                                        },
                                                        allowCross=False,
                                                        tooltip={
                                                            "always_visible": False
                                                        },
                                                    ),
                                                ],
                                                id="random-index-reg-pred-slider-div-"
                                                + self.name,
                                            ),
                                            html.Div(
                                                [
                                                    dbc.Label(
                                                        "Intervalo observado:", # Traduzido
                                                        id="random-index-reg-y-slider-label-"
                                                        + self.name,
                                                        html_for="random-index-reg-y-slider-"
                                                        + self.name,
                                                    ),
                                                    dbc.Tooltip(
                                                        f"Selecionar apenas {self.explainer.index_name} onde o "
                                                        f"{self.explainer.target} observado estava dentro do seguinte intervalo:", # Traduzido
                                                        target="random-index-reg-y-slider-label-"
                                                        + self.name,
                                                    ),
                                                    dcc.RangeSlider(
                                                        id="random-index-reg-y-slider-"
                                                        + self.name,
                                                        min=float(
                                                            self.explainer.y.min()
                                                        ),
                                                        max=float(
                                                            self.explainer.y.max()
                                                        ),
                                                        step=np.float_power(
                                                            10, -self.round
                                                        ),
                                                        value=[
                                                            self.y_slider[0],
                                                            self.y_slider[1],
                                                        ],
                                                        marks={
                                                            float(
                                                                self.explainer.y.min()
                                                            ): str(
                                                                np.round(
                                                                    self.explainer.y.min(),
                                                                    self.round,
                                                                )
                                                            ),
                                                            float(
                                                                self.explainer.y.max()
                                                            ): str(
                                                                np.round(
                                                                    self.explainer.y.max(),
                                                                    self.round,
                                                                )
                                                            ),
                                                        },
                                                        allowCross=False,
                                                        tooltip={
                                                            "always_visible": False
                                                        },
                                                    ),
                                                ],
                                                id="random-index-reg-y-slider-div-"
                                                + self.name,
                                            ),
                                        ],
                                        md=8,
                                    ),
                                    hide=self.hide_pred_slider,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Intervalo:", # Traduzido
                                                id="random-index-reg-preds-or-y-label-"
                                                + self.name,
                                                html_for="random-index-reg-preds-or-y-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="random-index-reg-preds-or-y-"
                                                + self.name,
                                                options=[
                                                    {
                                                        "label": "Previsto", # Traduzido
                                                        "value": "preds",
                                                    },
                                                    {"label": "Observado", "value": "y"}, # Traduzido
                                                ],
                                                value=self.pred_or_y,
                                            ),
                                            dbc.Tooltip(
                                                f"Pode selecionar um {self.explainer.index_name} aleatório apenas dentro de um certo intervalo do {self.explainer.target} observado ou dentro de um certo intervalo do {self.explainer.target} previsto.", # Traduzido
                                                target="random-index-reg-preds-or-y-label-"
                                                + self.name,
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    hide=self.hide_pred_or_y,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    dbc.Label(
                                                        "Intervalo de resíduos:", # Traduzido
                                                        id="random-index-reg-residual-slider-label-"
                                                        + self.name,
                                                        html_for="random-index-reg-residual-slider-"
                                                        + self.name,
                                                    ),
                                                    dbc.Tooltip(
                                                        f"Selecionar apenas {self.explainer.index_name} onde o "
                                                        f"resíduo (diferença entre o {self.explainer.target} observado e o {self.explainer.target} previsto)"
                                                        " estava dentro do seguinte intervalo:", # Traduzido
                                                        target="random-index-reg-residual-slider-label-"
                                                        + self.name,
                                                    ),
                                                    dcc.RangeSlider(
                                                        id="random-index-reg-residual-slider-"
                                                        + self.name,
                                                        min=float(
                                                            self.explainer.residuals.min()
                                                        ),
                                                        max=float(
                                                            self.explainer.residuals.max()
                                                        ),
                                                        step=float(
                                                            np.float_power(
                                                                10, -self.round
                                                            )
                                                        ),
                                                        value=[
                                                            float(
                                                                self.residual_slider[0]
                                                            ),
                                                            float(
                                                                self.residual_slider[1]
                                                            ),
                                                        ],
                                                        marks={
                                                            float(
                                                                self.explainer.residuals.min()
                                                            ): str(
                                                                np.round(
                                                                    self.explainer.residuals.min(),
                                                                    self.round,
                                                                )
                                                            ),
                                                            float(
                                                                self.explainer.residuals.max()
                                                            ): str(
                                                                np.round(
                                                                    self.explainer.residuals.max(),
                                                                    self.round,
                                                                )
                                                            ),
                                                        },
                                                        allowCross=False,
                                                        tooltip={
                                                            "always_visible": False
                                                        },
                                                    ),
                                                ],
                                                id="random-index-reg-residual-slider-div-"
                                                + self.name,
                                            ),
                                            html.Div(
                                                [
                                                    dbc.Label(
                                                        "Resíduos absolutos", # Traduzido
                                                        id="random-index-reg-abs-residual-slider-label"
                                                        + self.name,
                                                        html_for="random-index-reg-abs-residual-slider-"
                                                        + self.name,
                                                    ),
                                                    dbc.Tooltip(
                                                        f"Selecionar apenas {self.explainer.index_name} onde o resíduo absoluto "
                                                        f"(diferença entre o {self.explainer.target} observado e o {self.explainer.target} previsto)"
                                                        " estava dentro do seguinte intervalo:", # Traduzido
                                                        target="random-index-reg-abs-residual-slider-label"
                                                        + self.name,
                                                    ),
                                                    dcc.RangeSlider(
                                                        id="random-index-reg-abs-residual-slider-"
                                                        + self.name,
                                                        min=float(
                                                            self.explainer.abs_residuals.min()
                                                        ),
                                                        max=float(
                                                            self.explainer.abs_residuals.max()
                                                        ),
                                                        step=float(
                                                            np.float_power(
                                                                10, -self.round
                                                            )
                                                        ),
                                                        value=[
                                                            float(
                                                                self.abs_residual_slider[
                                                                    0
                                                                ]
                                                            ),
                                                            float(
                                                                self.abs_residual_slider[
                                                                    1
                                                                ]
                                                            ),
                                                        ],
                                                        marks={
                                                            float(
                                                                self.explainer.abs_residuals.min()
                                                            ): str(
                                                                np.round(
                                                                    self.explainer.abs_residuals.min(),
                                                                    self.round,
                                                                )
                                                            ),
                                                            float(
                                                                self.explainer.abs_residuals.max()
                                                            ): str(
                                                                np.round(
                                                                    self.explainer.abs_residuals.max(),
                                                                    self.round,
                                                                )
                                                            ),
                                                        },
                                                        allowCross=False,
                                                        tooltip={
                                                            "always_visible": False
                                                        },
                                                    ),
                                                ],
                                                id="random-index-reg-abs-residual-slider-div-"
                                                + self.name,
                                            ),
                                        ],
                                        md=8,
                                    ),
                                    hide=self.hide_residual_slider,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Resíduos:", # Traduzido
                                                id="random-index-reg-abs-residual-label-"
                                                + self.name,
                                                html_for="random-index-reg-abs-residual-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="random-index-reg-abs-residual-"
                                                + self.name,
                                                options=[
                                                    {
                                                        "label": "Resíduos", # Traduzido
                                                        "value": "relative",
                                                    },
                                                    {
                                                        "label": "Resíduos Absolutos", # Traduzido
                                                        "value": "absolute",
                                                    },
                                                ],
                                                value="absolute"
                                                if self.abs_residuals
                                                else "relative",
                                            ),
                                            dbc.Tooltip(
                                                f"Pode selecionar um {self.explainer.index_name} aleatório apenas "
                                                f"dentro de um certo intervalo de resíduos "
                                                f"(diferença entre o {self.explainer.target} observado e previsto), "
                                                f"por exemplo, apenas {self.explainer.index_name} para os quais a previsão "
                                                f"foi demasiado alta ou demasiado baixa."
                                                f"Ou pode selecionar apenas dentro de um certo intervalo de resíduos absolutos. Por "
                                                f"exemplo, selecionar apenas {self.explainer.index_name} para os quais a previsão estava errada "
                                                f"por pelo menos uma certa quantidade de {self.explainer.units}.", # Traduzido
                                                target="random-index-reg-abs-residual-label-"
                                                + self.name,
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    hide=self.hide_abs_residuals,
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        )

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)

        html = to_html.card(
            f"Índice selecionado: <b>{self.explainer.get_index(args['index'])}</b>", # Traduzido
            title=self.title,
        )
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        # Callbacks não contêm texto visível ao utilizador, exceto talvez mensagens de erro
        # que são difíceis de prever sem executar. As PreventUpdate não mostram nada.
        # As lógicas internas permanecem em inglês.
        @app.callback(
            [
                Output("random-index-reg-pred-slider-div-" + self.name, "style"),
                Output("random-index-reg-y-slider-div-" + self.name, "style"),
            ],
            [Input("random-index-reg-preds-or-y-" + self.name, "value")],
        )
        def update_reg_hidden_div_pred_sliders(preds_or_y):
            if preds_or_y == "preds":
                return (None, dict(display="none"))
            elif preds_or_y == "y":
                return (dict(display="none"), None)
            raise PreventUpdate

        @app.callback(
            [
                Output("random-index-reg-residual-slider-div-" + self.name, "style"),
                Output(
                    "random-index-reg-abs-residual-slider-div-" + self.name, "style"
                ),
            ],
            [Input("random-index-reg-abs-residual-" + self.name, "value")],
        )
        def update_reg_hidden_div_pred_sliders(abs_residuals):
            if abs_residuals == "absolute":
                return (dict(display="none"), None)
            else:
                return (None, dict(display="none"))
            raise PreventUpdate

        @app.callback(
            [
                Output("random-index-reg-residual-slider-" + self.name, "min"),
                Output("random-index-reg-residual-slider-" + self.name, "max"),
                Output("random-index-reg-residual-slider-" + self.name, "value"),
                Output("random-index-reg-residual-slider-" + self.name, "marks"),
                Output("random-index-reg-abs-residual-slider-" + self.name, "min"),
                Output("random-index-reg-abs-residual-slider-" + self.name, "max"),
                Output("random-index-reg-abs-residual-slider-" + self.name, "value"),
                Output("random-index-reg-abs-residual-slider-" + self.name, "marks"),
            ],
            [
                Input("random-index-reg-pred-slider-" + self.name, "value"),
                Input("random-index-reg-y-slider-" + self.name, "value"),
            ],
            [
                State("random-index-reg-preds-or-y-" + self.name, "value"),
                State("random-index-reg-residual-slider-" + self.name, "value"),
                State("random-index-reg-abs-residual-slider-" + self.name, "value"),
            ],
        )
        def update_residual_slider_limits(
            pred_range, y_range, preds_or_y, residuals_range, abs_residuals_range
        ):
            # Lógica interna, sem texto visível
            if preds_or_y == "preds":
                min_residuals = float(
                    self.explainer.residuals[
                        (self.explainer.preds >= pred_range[0])
                        & (self.explainer.preds <= pred_range[1])
                    ].min()
                )
                max_residuals = float(
                    self.explainer.residuals[
                        (self.explainer.preds >= pred_range[0])
                        & (self.explainer.preds <= pred_range[1])
                    ].max()
                )
                min_abs_residuals = float(
                    self.explainer.abs_residuals[
                        (self.explainer.preds >= pred_range[0])
                        & (self.explainer.preds <= pred_range[1])
                    ].min()
                )
                max_abs_residuals = float(
                    self.explainer.abs_residuals[
                        (self.explainer.preds >= pred_range[0])
                        & (self.explainer.preds <= pred_range[1])
                    ].max()
                )
            elif preds_or_y == "y":
                min_residuals = float(
                    self.explainer.residuals[
                        (self.explainer.y >= y_range[0])
                        & (self.explainer.y <= y_range[1])
                    ].min()
                )
                max_residuals = float(
                    self.explainer.residuals[
                        (self.explainer.y >= y_range[0])
                        & (self.explainer.y <= y_range[1])
                    ].max()
                )
                min_abs_residuals = float(
                    self.explainer.abs_residuals[
                        (self.explainer.y >= y_range[0])
                        & (self.explainer.y <= y_range[1])
                    ].min()
                )
                max_abs_residuals = float(
                    self.explainer.abs_residuals[
                        (self.explainer.y >= y_range[0])
                        & (self.explainer.y <= y_range[1])
                    ].max()
                )

            new_residuals_range = [
                max(min_residuals, residuals_range[0]),
                min(max_residuals, residuals_range[1]),
            ]
            new_abs_residuals_range = [
                max(min_abs_residuals, abs_residuals_range[0]),
                min(max_abs_residuals, abs_residuals_range[1]),
            ]
            residuals_marks = {
                min_residuals: str(np.round(min_residuals, self.round)),
                max_residuals: str(np.round(max_residuals, self.round)),
            }
            abs_residuals_marks = {
                min_abs_residuals: str(np.round(min_abs_residuals, self.round)),
                max_abs_residuals: str(np.round(max_abs_residuals, self.round)),
            }
            return (
                min_residuals,
                max_residuals,
                new_residuals_range,
                residuals_marks,
                min_abs_residuals,
                max_abs_residuals,
                new_abs_residuals_range,
                abs_residuals_marks,
            )

        @app.callback(
            Output("random-index-reg-index-" + self.name, "value"),
            [Input("random-index-reg-button-" + self.name, "n_clicks")],
            [
                State("random-index-reg-pred-slider-" + self.name, "value"),
                State("random-index-reg-y-slider-" + self.name, "value"),
                State("random-index-reg-residual-slider-" + self.name, "value"),
                State("random-index-reg-abs-residual-slider-" + self.name, "value"),
                State("random-index-reg-preds-or-y-" + self.name, "value"),
                State("random-index-reg-abs-residual-" + self.name, "value"),
            ],
        )
        def update_index(
            n_clicks,
            pred_range,
            y_range,
            residual_range,
            abs_residuals_range,
            preds_or_y,
            abs_residuals,
        ):
            # Lógica interna, sem texto visível
            triggers = [
                trigger["prop_id"] for trigger in dash.callback_context.triggered
            ]
            if f"random-index-reg-button-{self.name}.n_clicks" not in triggers:
                raise PreventUpdate
            if n_clicks is None and self.index is not None:
                raise PreventUpdate
            if preds_or_y == "preds":
                if abs_residuals == "absolute":
                    return self.explainer.random_index(
                        pred_min=pred_range[0],
                        pred_max=pred_range[1],
                        abs_residuals_min=abs_residuals_range[0],
                        abs_residuals_max=abs_residuals_range[1],
                        return_str=True,
                    )
                else:
                    return self.explainer.random_index(
                        pred_min=pred_range[0],
                        pred_max=pred_range[1],
                        residuals_min=residual_range[0],
                        residuals_max=residual_range[1],
                        return_str=True,
                    )
            elif preds_or_y == "y":
                if abs_residuals == "absolute":
                    return self.explainer.random_index(
                        y_min=y_range[0],
                        y_max=y_range[1],
                        abs_residuals_min=abs_residuals_range[0],
                        abs_residuals_max=abs_residuals_range[1],
                        return_str=True,
                    )
                else:
                    return self.explainer.random_index(
                        y_min=pred_range[0], # Atenção: parece haver um erro aqui no código original, deveria ser y_range? Mantido como no original.
                        y_max=pred_range[1], # Atenção: parece haver um erro aqui no código original, deveria ser y_range? Mantido como no original.
                        residuals_min=residual_range[0],
                        residuals_max=residual_range[1],
                        return_str=True,
                    )


class RegressionModelSummaryComponent(ExplainerComponent):
    def __init__(
        self,
        explainer,
        title="Sumário do Modelo", # Traduzido
        name=None,
        subtitle="Métricas quantitativas para o desempenho do modelo", # Traduzido
        hide_title=False,
        hide_subtitle=False,
        round=3,
        show_metrics=None,
        description=None,
        **kwargs,
    ):
        """Componente que mostra estatísticas de sumário do modelo (RMSE, MAE, R2)

        Args:
            explainer (Explainer): objeto explainer construído com
                        ClassifierExplainer() ou RegressionExplainer()
            title (str, optional): Título do separador ou página. Predefinição para
                        "Sumário do Modelo".
            name (str, optional): nome único a adicionar aos elementos do Componente.
                        Se None, um uuid aleatório é gerado para garantir
                        que é único. Predefinição para None.
            subtitle (str): subtítulo
            hide_title (bool, optional): ocultar título
            hide_subtitle (bool, optional): Ocultar subtítulo. Predefinição para False.
            round (int): arredondamento a aplicar às métricas de ponto flutuante.
            show_metrics (List): lista de métricas a exibir por ordem. Predefinição
                para None, exibindo todas as métricas.
            description (str, optional): Dica a exibir ao passar o rato sobre
                o título do componente. Quando None, o texto predefinido é mostrado.
        """
        super().__init__(explainer, title, name)
        if self.description is None:
            self.description = f"""
        Na tabela abaixo pode encontrar várias métricas de desempenho de regressão
        que descrevem quão bem o modelo consegue prever
        {self.explainer.target}.
        """ # Traduzido
        self.register_dependencies(["preds", "residuals"])

    def layout(self):
        metrics_dict = self.explainer.metrics_descriptions()
        metrics_df = (
            pd.DataFrame(
                self.explainer.metrics(show_metrics=self.show_metrics), index=["Valor"] # Traduzido 'Score'
            )
            .T.rename_axis(index="métrica") # Traduzido 'metric'
            .reset_index()
            .round(self.round)
        )
        # Nota: Os nomes das métricas (RMSE, MAE, etc.) vêm de self.explainer.metrics()
        # e geralmente são mantidos como acrónimos. Se precisarem de tradução,
        # teria de ser feito dentro do método `metrics` ou `metrics_descriptions`.
        metrics_table = dbc.Table.from_dataframe(
            metrics_df, striped=False, bordered=False, hover=False
        )
        # As tooltips vêm de metrics_dict, que não está definido aqui, mas assume-se que
        # as descrições nesse dicionário precisariam de tradução se fossem exibidas.
        metrics_table, tooltips = get_dbc_tooltips(
            metrics_table, metrics_dict, "reg-model-summary-div-hover", self.name
        )
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title, # Já traduzido
                                        id="reg-model-summary-title-" + self.name,
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle" # Já traduzido
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description, # Já traduzido
                                        target="reg-model-summary-title-" + self.name,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    hide=self.hide_title,
                ),
                dbc.CardBody([metrics_table, *tooltips]),
            ]
        )

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        metrics_df = (
            pd.DataFrame(
                self.explainer.metrics(show_metrics=self.show_metrics), index=["Valor"] # Traduzido 'Score'
            )
            .T.rename_axis(index="métrica") # Traduzido 'metric'
            .reset_index()
            .round(self.round)
        )
        html = to_html.table_from_df(metrics_df)
        html = to_html.card(html, title=self.title)
        if add_header:
            return to_html.add_header(html)
        return html


class RegressionPredictionSummaryComponent(ExplainerComponent):
    _state_props = dict(index=("reg-prediction-index-", "value"))

    def __init__(
        self,
        explainer,
        title="Previsão", # Traduzido
        name=None,
        hide_index=False,
        hide_title=False,
        hide_subtitle=False, # Subtítulo não usado neste componente por defeito
        hide_table=False,
        index_dropdown=True,
        feature_input_component=None,
        index=None,
        round=3,
        description=None,
        **kwargs,
    ):
        """Mostra um sumário para uma previsão específica

        Args:
            explainer (Explainer): objeto explainer construído com
                        ClassifierExplainer() ou RegressionExplainer()
            title (str, optional): Título do separador ou página. Predefinição para
                        "Previsão".
            name (str, optional): nome único a adicionar aos elementos do Componente.
                        Se None, um uuid aleatório é gerado para garantir
                        que é único. Predefinição para None.
            hide_index (bool, optional): ocultar seletor de índice. Predefinição para False.
            hide_title (bool, optional): ocultar título. Predefinição para False.
            hide_subtitle (bool, optional): Ocultar subtítulo. Predefinição para False.
            hide_table (bool, optional): ocultar a tabela de resultados
            index_dropdown (bool, optional): Usar lista pendente para entrada de índice em vez
                de entrada de texto livre. Predefinição para True.
            feature_input_component (FeatureInputComponent): Um FeatureInputComponent
                que fornecerá a entrada para o gráfico em vez do seletor de índice.
                Se não for None, hide_index=True. Predefinição para None.
            index ({int, str}, optional): Índice para exibir o sumário da previsão. Predefinição para None.
            description (str, optional): Dica a exibir ao passar o rato sobre
                o título do componente. Quando None, o texto predefinido é mostrado.
        """
        super().__init__(explainer, title, name)

        self.index_name = "reg-prediction-index-" + self.name
        self.index_selector = IndexSelector(
            explainer, self.index_name, index=index, index_dropdown=index_dropdown
        )

        if self.feature_input_component is not None:
            self.exclude_callbacks(self.feature_input_component)
            self.hide_index = True

        if self.description is None:
            self.description = f"""
        Mostra o {self.explainer.target} previsto e o {self.explainer.target} observado,
        bem como a diferença entre os dois (resíduo).
        """ # Traduzido

    def layout(self):
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.H3(
                                self.title, # Já traduzido
                                id="reg-prediction-title-" + self.name,
                                className="card-title",
                            ),
                            dbc.Tooltip(
                                self.description, # Já traduzido
                                target="reg-prediction-title-" + self.name,
                            ),
                        ]
                    ),
                    hide=self.hide_title,
                ),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(f"{self.explainer.index_name}:"), # Mantém nome da variável
                                            self.index_selector.layout(),
                                        ],
                                        md=6,
                                    ),
                                    hide=self.hide_index,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [dbc.Col([html.Div(id="reg-prediction-div-" + self.name)])]
                        ),
                    ]
                ),
            ]
        )

    def get_state_tuples(self):
        _state_tuples = super().get_state_tuples()
        if self.feature_input_component is not None:
            _state_tuples.extend(self.feature_input_component.get_state_tuples())
        return sorted(list(set(_state_tuples)))

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        if self.feature_input_component is None:
            if args["index"] is not None:
                # Assume que prediction_result_df retorna um df com colunas como
                # 'Predicted', 'Actual', 'Residual'. A tradução ocorreria DENTRO
                # dessa função ou seria aplicada aqui se os nomes fossem fixos.
                # Exemplo: preds_df.columns = ["Previsto", "Observado", "Resíduo"]
                preds_df = self.explainer.prediction_result_df(
                    args["index"], round=self.round
                )
                # Para demonstração, vamos assumir que a função retorna colunas padrão
                # e traduzimos aqui (isto pode não ser o ideal na prática)
                try:
                    preds_df = preds_df.rename(columns={'Predicted': 'Previsto', 'Actual': 'Observado', 'Residual': 'Resíduo'})
                except: # Ignora se as colunas não existirem
                    pass
                html = to_html.table_from_df(preds_df)
            else:
                html = "nenhum índice selecionado" # Traduzido
        else:
            inputs = {
                k: v
                for k, v in self.feature_input_component.get_state_args(
                    state_dict
                ).items()
                if k != "index"
            }
            inputs = list(inputs.values())
            if len(inputs) == len(
                self.feature_input_component._input_features
            ) and not any([i is None for i in inputs]):
                X_row = self.explainer.get_row_from_input(inputs, ranked_by_shap=True)
                preds_df = self.explainer.prediction_result_df(
                    X_row=X_row, round=self.round
                )
                # Tradução das colunas como acima
                try:
                    preds_df = preds_df.rename(columns={'Predicted': 'Previsto', 'Actual': 'Observado', 'Residual': 'Resíduo'})
                except:
                    pass
                html = to_html.table_from_df(preds_df)
            else:
                html = f"<div>dados de entrada incorretos</div>" # Traduzido

        html = to_html.card(html, title=self.title)
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        if self.feature_input_component is None:

            @app.callback(
                Output("reg-prediction-div-" + self.name, "children"),
                [Input("reg-prediction-index-" + self.name, "value")],
            )
            def update_output_div(index):
                if index is None or not self.explainer.index_exists(index):
                    raise PreventUpdate
                preds_df = self.explainer.prediction_result_df(index, round=self.round)
                # Tradução das colunas como acima
                try:
                    preds_df = preds_df.rename(columns={'Predicted': 'Previsto', 'Actual': 'Observado', 'Residual': 'Resíduo'})
                except:
                    pass
                return make_hideable(
                    dbc.Table.from_dataframe(
                        preds_df, striped=False, bordered=False, hover=False
                    ),
                    hide=self.hide_table,
                )

        else:

            @app.callback(
                Output("reg-prediction-div-" + self.name, "children"),
                [*self.feature_input_component._feature_callback_inputs],
            )
            def update_output_div(*inputs):
                X_row = self.explainer.get_row_from_input(inputs, ranked_by_shap=True)
                preds_df = self.explainer.prediction_result_df(
                    X_row=X_row, round=self.round
                )
                # Tradução das colunas como acima
                try:
                    preds_df = preds_df.rename(columns={'Predicted': 'Previsto', 'Actual': 'Observado', 'Residual': 'Resíduo'})
                except:
                    pass
                return make_hideable(
                    dbc.Table.from_dataframe(
                        preds_df, striped=False, bordered=False, hover=False
                    ),
                    hide=self.hide_table,
                )


class PredictedVsActualComponent(ExplainerComponent):
    _state_props = dict(
        log_x=("pred-vs-actual-logx-", "value"), log_y=("pred-vs-actual-logy-", "value")
    )

    def __init__(
        self,
        explainer,
        title="Previsto vs. Observado", # Traduzido
        name=None,
        subtitle="Quão próximo está o valor previsto do observado?", # Traduzido
        hide_title=False,
        hide_subtitle=False,
        hide_log_x=False,
        hide_log_y=False,
        hide_popout=False,
        logs=False,
        log_x=False,
        log_y=False,
        round=3,
        plot_sample=None,
        description=None,
        **kwargs,
    ):
        """Mostra um gráfico de previsões vs y.

        Args:
            explainer (Explainer): objeto explainer construído com
                        ClassifierExplainer() ou RegressionExplainer()
            title (str, optional): Título do separador ou página. Predefinição para
                        "Previsto vs. Observado".
            name (str, optional): nome único a adicionar aos elementos do Componente.
                        Se None, um uuid aleatório é gerado para garantir
                        que é único. Predefinição para None.
            subtitle (str): subtítulo
            hide_title (bool, optional) Ocultar o título. Predefinição para False.
            hide_subtitle (bool, optional): Ocultar subtítulo. Predefinição para False.
            hide_log_x (bool, optional): Ocultar o alternador log_x. Predefinição para False.
            hide_log_y (bool, optional): Ocultar o alternador log_y. Predefinição para False.
            hide_popout (bool, optional): ocultar botão popout. Predefinição para False.
            logs (bool, optional): Se deve usar eixo logarítmico. Predefinição para False.
            log_x (bool, optional): log apenas no eixo x. Predefinição para False.
            log_y (bool, optional): log apenas no eixo y. Predefinição para False.
            round (int, optional): arredondamento a aplicar às previsões de ponto flutuante.
                Predefinição para 3.
            plot_sample (int, optional): Em vez de todos os pontos, plotar apenas uma amostra
                aleatória de pontos. Predefinição para None (=todos os pontos)
            description (str, optional): Dica a exibir ao passar o rato sobre
                o título do componente. Quando None, o texto predefinido é mostrado.
        """
        super().__init__(explainer, title, name)

        self.logs, self.log_x, self.log_y = logs, log_x, log_y

        if self.description is None:
            self.description = f"""
        O gráfico mostra o {self.explainer.target} observado e o {self.explainer.target} previsto
        no mesmo gráfico. Um modelo perfeito teria
        todos os pontos na diagonal (previsto igual a observado). Quanto mais
        afastados os pontos estiverem da diagonal, pior é o modelo a prever
        {self.explainer.target}.
        """ # Traduzido

        self.popout = GraphPopout(
            "pred-vs-actual-" + self.name + "popout",
            "pred-vs-actual-graph-" + self.name,
            self.title,
            self.description,
        )
        self.register_dependencies(["preds"])

    def layout(self):
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title, # Já traduzido
                                        id="pred-vs-actual-title-" + self.name,
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle" # Já traduzido
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description, # Já traduzido
                                        target="pred-vs-actual-title-" + self.name,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    hide=self.hide_title,
                ),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Row(
                                                [
                                                    # html.Label("Log y"), # Comentado no original
                                                    dbc.RadioButton(
                                                        id="pred-vs-actual-logy-"
                                                        + self.name,
                                                        className="form-check-input",
                                                        value=self.log_y,
                                                    ),
                                                    dbc.Tooltip(
                                                        "Ao usar um eixo logarítmico, é mais fácil ver erros relativos "
                                                        "em vez de erros absolutos.", # Traduzido
                                                        target="pred-vs-actual-logy-"
                                                        + self.name,
                                                    ),
                                                    dbc.Label(
                                                        "Log y", # Mantido "Log"
                                                        html_for="pred-vs-actual-logy-"
                                                        + self.name,
                                                        className="form-check-label",
                                                        size="sm",
                                                    ),
                                                ]
                                            ),
                                        ],
                                        md=1,
                                        align="center",
                                    ),
                                    hide=self.hide_log_y,
                                ),
                                dbc.Col(
                                    [
                                        dcc.Graph(
                                            id="pred-vs-actual-graph-" + self.name,
                                            config=dict(
                                                modeBarButtons=[["toImage"]],
                                                displaylogo=False,
                                            ),
                                        ),
                                    ],
                                    md=11,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.RadioButton(
                                                        id="pred-vs-actual-logx-"
                                                        + self.name,
                                                        className="form-check-input",
                                                        value=self.log_x,
                                                    ),
                                                    dbc.Tooltip(
                                                        "Ao usar um eixo logarítmico, é mais fácil ver erros relativos "
                                                        "em vez de erros absolutos.", # Traduzido
                                                        target="pred-vs-actual-logx-"
                                                        + self.name,
                                                    ),
                                                    dbc.Label(
                                                        "Log x", # Mantido "Log"
                                                        html_for="pred-vs-actual-logx-"
                                                        + self.name,
                                                        className="form-check-label",
                                                        size="sm",
                                                    ),
                                                ]
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    hide=self.hide_log_x,
                                ),
                            ],
                            justify="center",
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [self.popout.layout()], md=2, align="start"
                                    ),
                                    hide=self.hide_popout,
                                ),
                            ],
                            justify="end",
                        ),
                    ]
                ),
            ]
        )

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        # Nota: plot_predicted_vs_actual pode precisar de ter os seus eixos traduzidos
        # internamente ou os rótulos podem ser sobrescritos aqui.
        # Ex: fig.update_layout(xaxis_title="Observado", yaxis_title="Previsto")
        fig = self.explainer.plot_predicted_vs_actual(
            log_x=bool(args["log_x"]),
            log_y=bool(args["log_y"]),
            round=self.round,
            plot_sample=self.plot_sample,
        )
        # Exemplo de tradução de eixos (se necessário):
        fig.update_layout(xaxis_title=f"{self.explainer.target} Observado", yaxis_title=f"{self.explainer.target} Previsto")
        html = to_html.card(to_html.fig(fig), title=self.title)
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        @app.callback(
            Output("pred-vs-actual-graph-" + self.name, "figure"),
            [
                Input("pred-vs-actual-logx-" + self.name, "value"),
                Input("pred-vs-actual-logy-" + self.name, "value"),
            ],
        )
        def update_predicted_vs_actual_graph(log_x, log_y):
            fig = self.explainer.plot_predicted_vs_actual(
                log_x=log_x, log_y=log_y, round=self.round, plot_sample=self.plot_sample
            )
            # Exemplo de tradução de eixos (se necessário):
            fig.update_layout(xaxis_title=f"{self.explainer.target} Observado", yaxis_title=f"{self.explainer.target} Previsto")
            return fig


class ResidualsComponent(ExplainerComponent):
    _state_props = dict(
        pred_or_actual=("residuals-pred-or-actual-", "value"),
        residuals=("residuals-type-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Resíduos", # Traduzido
        name=None,
        subtitle="Qual o desvio do modelo?", # Traduzido
        hide_title=False,
        hide_subtitle=False,
        hide_footer=False,
        hide_pred_or_actual=False,
        hide_ratio=False,
        hide_popout=False,
        pred_or_actual="vs_pred",
        residuals="difference",
        round=3,
        plot_sample=None,
        description=None,
        **kwargs,
    ):
        """Componente de gráfico de resíduos

        Args:
            explainer (Explainer): objeto explainer construído com
                        ClassifierExplainer() ou RegressionExplainer()
            title (str, optional): Título do separador ou página. Predefinição para
                        "Resíduos".
            name (str, optional): nome único a adicionar aos elementos do Componente.
                        Se None, um uuid aleatório é gerado para garantir
                        que é único. Predefinição para None.
            subtitle (str): subtítulo
            hide_title (bool, optional) Ocultar o título. Predefinição para False.
            hide_subtitle (bool, optional): Ocultar subtítulo. Predefinição para False.
            hide_footer (bool, optional): ocultar o rodapé na parte inferior do componente
            hide_pred_or_actual (bool, optional): ocultar vs previsões ou vs
                        real para alternador do eixo x. Predefinição para False.
            hide_ratio (bool, optional): ocultar lista pendente do tipo de resíduo. Predefinição para False.
            hide_popout (bool, optional): ocultar botão popout. Predefinição para False.
            pred_or_actual (str, {'vs_actual', 'vs_pred'}, optional): Se deve
                        plotar real ou previsões no eixo x.
                        Predefinição para "vs_pred".
            residuals (str, {'difference', 'ratio', 'log-ratio'} optional):
                    Como calcular resíduos. Predefinição para 'difference'.
            round (int, optional): arredondamento a aplicar às previsões de ponto flutuante.
                Predefinição para 3.
            plot_sample (int, optional): Em vez de todos os pontos, plotar apenas uma amostra
                aleatória de pontos. Predefinição para None (=todos os pontos)
            description (str, optional): Dica a exibir ao passar o rato sobre
                o título do componente. Quando None, o texto predefinido é mostrado.
        """
        super().__init__(explainer, title, name)

        assert residuals in ["difference", "ratio", "log-ratio"], (
            "o parâmetro residuals deve estar em ['difference', 'ratio', 'log-ratio']"
            f" mas passou residuals={residuals}"
        )

        if self.description is None:
            self.description = f"""
        Os resíduos são a diferença entre o {self.explainer.target} observado
        e o {self.explainer.target} previsto. Neste gráfico, pode verificar se
        os resíduos são maiores ou menores para resultados observados/previstos mais altos/baixos.
        Assim, pode verificar se o modelo funciona melhor ou worse para diferentes níveis de {self.explainer.target}.
        """ # Traduzido

        self.popout = GraphPopout(
            "residuals-" + self.name + "popout",
            "residuals-graph-" + self.name,
            self.title,
            self.description,
        )
        self.register_dependencies(["preds", "residuals"])

    def layout(self):
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title, id="residuals-title-" + self.name # Já traduzido
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle" # Já traduzido
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description, # Já traduzido
                                        target="residuals-title-" + self.name,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    hide=self.hide_title,
                ),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dcc.Graph(
                                            id="residuals-graph-" + self.name,
                                            config=dict(
                                                modeBarButtons=[["toImage"]],
                                                displaylogo=False,
                                            ),
                                        ),
                                    ]
                                )
                            ]
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [self.popout.layout()], md=2, align="start"
                                    ),
                                    hide=self.hide_popout,
                                ),
                            ],
                            justify="end",
                        ),
                    ]
                ),
                make_hideable(
                    dbc.CardFooter(
                        [
                            dbc.Row(
                                [
                                    make_hideable(
                                        dbc.Col(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Label(
                                                            "Eixo horizontal:", # Traduzido
                                                            html_for="residuals-pred-or-actual-"
                                                            + self.name,
                                                        ),
                                                        dbc.Select(
                                                            options=[
                                                                {
                                                                    "label": "Previsto", # Traduzido
                                                                    "value": "vs_pred",
                                                                },
                                                                {
                                                                    "label": "Observado", # Traduzido
                                                                    "value": "vs_actual",
                                                                },
                                                            ],
                                                            value=self.pred_or_actual,
                                                            id="residuals-pred-or-actual-"
                                                            + self.name,
                                                            size="sm",
                                                        ),
                                                    ],
                                                    id="residuals-pred-or-actual-form-"
                                                    + self.name,
                                                ),
                                                dbc.Tooltip(
                                                    "Selecione o que gostaria de colocar no eixo x:"
                                                    f" {self.explainer.target} observado ou {self.explainer.target} previsto.", # Traduzido
                                                    target="residuals-pred-or-actual-form-"
                                                    + self.name,
                                                ),
                                            ],
                                            md=3,
                                        ),
                                        hide=self.hide_pred_or_actual,
                                    ),
                                    make_hideable(
                                        dbc.Col(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Label(
                                                            "Tipo de resíduo:", # Traduzido
                                                            id="residuals-type-label-"
                                                            + self.name,
                                                            html_for="residuals-type-"
                                                            + self.name,
                                                        ),
                                                        dbc.Tooltip(
                                                            "Tipo de resíduos a exibir: y-previsto (diferença), "
                                                            "y/previsto (rácio) ou log(y/previsto) (log-rácio).", # Traduzido
                                                            target="residuals-type-label-"
                                                            + self.name,
                                                        ),
                                                        dbc.Select(
                                                            id="residuals-type-"
                                                            + self.name,
                                                            options=[
                                                                {
                                                                    "label": "Diferença", # Traduzido
                                                                    "value": "difference",
                                                                },
                                                                {
                                                                    "label": "Rácio", # Traduzido
                                                                    "value": "ratio",
                                                                },
                                                                {
                                                                    "label": "Log-Rácio", # Traduzido
                                                                    "value": "log-ratio",
                                                                },
                                                            ],
                                                            value=self.residuals,
                                                            size="sm",
                                                        ),
                                                    ]
                                                ),
                                            ],
                                            md=3,
                                        ),
                                        hide=self.hide_ratio,
                                    ),
                                ],
                                justify="evenly",
                            ),
                        ]
                    ),
                    hide=self.hide_footer,
                ),
            ]
        )

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        vs_actual = args["pred_or_actual"] == "vs_actual"
        # Nota: plot_residuals pode precisar de ter os seus eixos traduzidos
        fig = self.explainer.plot_residuals(
            vs_actual=vs_actual,
            residuals=args["residuals"],
            round=self.round,
            plot_sample=self.plot_sample,
        )
        # Exemplo de tradução de eixos (se necessário):
        x_title = f"{self.explainer.target} Observado" if vs_actual else f"{self.explainer.target} Previsto"
        y_title_dict = {"difference": "Resíduo (Diferença)", "ratio": "Resíduo (Rácio)", "log-ratio": "Resíduo (Log-Rácio)"}
        y_title = y_title_dict.get(args["residuals"], "Resíduo")
        fig.update_layout(xaxis_title=x_title, yaxis_title=y_title, margin=dict(t=40, b=40, l=40, r=40))
        html = to_html.card(to_html.fig(fig), title=self.title)
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        @app.callback(
            Output("residuals-graph-" + self.name, "figure"),
            [
                Input("residuals-pred-or-actual-" + self.name, "value"),
                Input("residuals-type-" + self.name, "value"),
            ],
        )
        def update_residuals_graph(pred_or_actual, residuals):
            vs_actual = pred_or_actual == "vs_actual"
            fig = self.explainer.plot_residuals(
                vs_actual=vs_actual,
                residuals=residuals,
                round=self.round,
                plot_sample=self.plot_sample,
            )
            # Exemplo de tradução de eixos (se necessário):
            x_title = f"{self.explainer.target} Observado" if vs_actual else f"{self.explainer.target} Previsto"
            y_title_dict = {"difference": "Resíduo (Diferença)", "ratio": "Resíduo (Rácio)", "log-ratio": "Resíduo (Log-Rácio)"}
            y_title = y_title_dict.get(residuals, "Resíduo")
            fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
            return fig


class RegressionVsColComponent(ExplainerComponent):
    _state_props = dict(
        col=("reg-vs-col-col-", "value"),
        display=("reg-vs-col-display-type-", "value"),
        points=("reg-vs-col-show-points-", "value"),
        winsor=("reg-vs-col-winsor-", "value"),
        cats_topx=("reg-vs-col-n-categories-", "value"),
        cats_sort=("reg-vs-col-categories-sort-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Gráfico vs característica", # Traduzido
        name=None,
        subtitle="As previsões e os resíduos estão correlacionados com as características?", # Traduzido
        hide_title=False,
        hide_subtitle=False,
        hide_footer=False,
        hide_col=False,
        hide_ratio=False,
        hide_points=False,
        hide_winsor=False,
        hide_cats_topx=False,
        hide_cats_sort=False,
        hide_popout=False,
        col=None,
        display="difference",
        round=3,
        points=True,
        winsor=0,
        cats_topx=10,
        cats_sort="freq",
        plot_sample=None,
        description=None,
        **kwargs,
    ):
        """Componente que mostra resíduos, observados ou previstos vs uma Característica específica

        Args:
            explainer (Explainer): objeto explainer construído com
                        ClassifierExplainer() ou RegressionExplainer()
            title (str, optional): Título do separador ou página. Predefinição para
                        "Gráfico vs característica".
            name (str, optional): nome único a adicionar aos elementos do Componente.
                        Se None, um uuid aleatório é gerado para garantir
                        que é único. Predefinição para None.
            subtitle (str): subtítulo
            hide_title (bool, optional) Ocultar o título. Predefinição para False.
            hide_subtitle (bool, optional): Ocultar subtítulo. Predefinição para False.
            hide_footer (bool, optional): ocultar o rodapé na parte inferior do componente
            hide_col (bool, optional): Ocultar o seletor de coluna. Predefinição para False.
            hide_ratio (bool, optional): Ocultar o alternador. Predefinição para False.
            hide_points (bool, optional): Ocultar alternador de pontos de grupo. Predefinição para False.
            hide_winsor (bool, optional): Ocultar entrada winsor. Predefinição para False.
            hide_cats_topx (bool, optional): ocultar a entrada topx de categorias. Predefinição para False.
            hide_cats_sort (bool, optional): ocultar o seletor de ordenação de categorias. Predefinição para False.
            hide_popout (bool, optional): ocultar botão popout. Predefinição para False.
            col ([type], optional): Característica inicial a exibir. Predefinição para None.
            display (str, {'observed', 'predicted', difference', 'ratio', 'log-ratio'} optional):
                    O que exibir no eixo y. Predefinição para 'difference'.
            round (int, optional): arredondamento a aplicar às previsões de ponto flutuante.
                Predefinição para 3.
            points (bool, optional): exibir nuvem de pontos ao lado do gráfico de violino
                    para colunas categóricas. Predefinição para True
            winsor (int, 0-50, optional): percentagem de outliers a remover (winsorizar)
                    do eixo y. Predefinição para 0.
            cats_topx (int, optional): número máximo de categorias a exibir
                para características categóricas. Predefinição para 10.
            cats_sort (str, optional): como ordenar categorias: 'alphabet',
                'freq' ou 'shap'. Predefinição para 'freq'.
            plot_sample (int, optional): Em vez de todos os pontos, plotar apenas uma amostra
                aleatória de pontos. Predefinição para None (=todos os pontos)
            description (str, optional): Dica a exibir ao passar o rato sobre
                o título do componente. Quando None, o texto predefinido é mostrado.
        """
        super().__init__(explainer, title, name)

        if self.col is None:
            # Assume que columns_ranked_by_shap() retorna nomes de colunas que não precisam de tradução
            self.col = self.explainer.columns_ranked_by_shap()[0]

        assert self.display in {
            "observed",
            "predicted",
            "difference",
            "ratio",
            "log-ratio",
        }, (
            "o parâmetro display deve estar em {'observed', 'predicted', 'difference', 'ratio', 'log-ratio'}"
            f" mas passou display={self.display}!"
        )

        if self.description is None:
            self.description = f"""
        Este gráfico mostra os resíduos (diferença entre o {self.explainer.target} observado
        e o {self.explainer.target} previsto) plotados contra os valores de diferentes características,
        ou o {self.explainer.target} observado ou previsto.
        Isto permite inspecionar se o modelo está mais errado para intervalos específicos
        de valores de características do que para outros.
        """ # Traduzido
        self.popout = GraphPopout(
            "reg-vs-col-" + self.name + "popout",
            "reg-vs-col-graph-" + self.name,
            self.title,
            self.description,
        )
        self.register_dependencies(["preds", "residuals"])

    def layout(self):
        # Assume que columns_ranked_by_shap() retorna nomes de colunas que não precisam de tradução
        col_options = [{"label": col, "value": col} for col in self.explainer.columns_ranked_by_shap()]

        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title, id="reg-vs-col-title-" + self.name # Já traduzido
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle" # Já traduzido
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description, # Já traduzido
                                        target="reg-vs-col-title-" + self.name,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    hide=self.hide_title,
                ),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Característica:", # Traduzido
                                                id="reg-vs-col-col-label-" + self.name,
                                            ),
                                            dbc.Tooltip(
                                                "Selecione a característica a exibir no eixo x.", # Traduzido
                                                target="reg-vs-col-col-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="reg-vs-col-col-" + self.name,
                                                options=col_options, # Nomes das colunas não traduzidos
                                                value=self.col,
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    hide=self.hide_col,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Exibir:", # Traduzido
                                                id="reg-vs-col-display-type-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                f"Selecione o que exibir no eixo y: {self.explainer.target} observado, "
                                                f"{self.explainer.target} previsto ou resíduos. Os resíduos podem ser "
                                                "calculados pela diferença (y-previsto), "
                                                "rácio (y/previsto) ou log-rácio log(y/previsto). Este último facilita a "
                                                "visualização de diferenças relativas.", # Traduzido
                                                target="reg-vs-col-display-type-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="reg-vs-col-display-type-"
                                                + self.name,
                                                options=[
                                                    {
                                                        "label": "Observado", # Traduzido
                                                        "value": "observed",
                                                    },
                                                    {
                                                        "label": "Previsto", # Traduzido
                                                        "value": "predicted",
                                                    },
                                                    {
                                                        "label": "Resíduos: Diferença", # Traduzido
                                                        "value": "difference",
                                                    },
                                                    {
                                                        "label": "Resíduos: Rácio", # Traduzido
                                                        "value": "ratio",
                                                    },
                                                    {
                                                        "label": "Resíduos: Log-Rácio", # Traduzido
                                                        "value": "log-ratio",
                                                    },
                                                ],
                                                value=self.display,
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    hide=self.hide_ratio,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dcc.Graph(
                                            id="reg-vs-col-graph-" + self.name,
                                            config=dict(
                                                modeBarButtons=[["toImage"]],
                                                displaylogo=False,
                                            ),
                                        ),
                                    ]
                                )
                            ]
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [self.popout.layout()], md=2, align="start"
                                    ),
                                    hide=self.hide_popout,
                                ),
                            ],
                            justify="end",
                        ),
                    ]
                ),
                make_hideable(
                    dbc.CardFooter(
                        [
                            dbc.Row(
                                [
                                    make_hideable(
                                        dbc.Col(
                                            [
                                                dbc.Label(
                                                    "Winsor:", # Mantido termo técnico
                                                    id="reg-vs-col-winsor-label-"
                                                    + self.name,
                                                ),
                                                dbc.Tooltip(
                                                    "Exclui os valores y mais altos e mais baixos do gráfico. "
                                                    "Quando existem alguns outliers reais, pode ajudar removê-los"
                                                    " do gráfico para que seja mais fácil ver o padrão geral.", # Traduzido
                                                    target="reg-vs-col-winsor-label-"
                                                    + self.name,
                                                ),
                                                dbc.Input(
                                                    id="reg-vs-col-winsor-" + self.name,
                                                    value=self.winsor,
                                                    type="number",
                                                    min=0,
                                                    max=49,
                                                    step=1,
                                                ),
                                            ],
                                            md=4,
                                        ),
                                        hide=self.hide_winsor,
                                    ),
                                    make_hideable(
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    [
                                                        dbc.Row(
                                                            [
                                                                dbc.Label("Dispersão:"), # Traduzido 'Scatter'
                                                                dbc.Tooltip(
                                                                    "Para características categóricas, exibir "
                                                                    "uma nuvem de pontos ao lado dos gráficos de violino.", # Traduzido
                                                                    target="reg-vs-col-show-points-"
                                                                    + self.name,
                                                                ),
                                                                dbc.Checklist(
                                                                    options=[
                                                                        {
                                                                            "label": "Mostrar nuvem de pontos", # Traduzido
                                                                            "value": True,
                                                                        }
                                                                    ],
                                                                    value=[True]
                                                                    if self.points
                                                                    else [],
                                                                    id="reg-vs-col-show-points-"
                                                                    + self.name,
                                                                    inline=True,
                                                                    switch=True,
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                    id="reg-vs-col-show-points-div-"
                                                    + self.name,
                                                )
                                            ],
                                            md=2,
                                        ),
                                        self.hide_points,
                                    ),
                                    make_hideable(
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    [
                                                        dbc.Label(
                                                            "Categorias:", # Traduzido
                                                            id="reg-vs-col-n-categories-label-"
                                                            + self.name,
                                                        ),
                                                        dbc.Tooltip(
                                                            "Número máximo de categorias a exibir", # Traduzido
                                                            target="reg-vs-col-n-categories-label-"
                                                            + self.name,
                                                        ),
                                                        dbc.Input(
                                                            id="reg-vs-col-n-categories-"
                                                            + self.name,
                                                            value=self.cats_topx,
                                                            type="number",
                                                            min=1,
                                                            max=50,
                                                            step=1,
                                                        ),
                                                    ],
                                                    id="reg-vs-col-n-categories-div-"
                                                    + self.name,
                                                ),
                                            ],
                                            md=2,
                                        ),
                                        self.hide_cats_topx,
                                    ),
                                    make_hideable(
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    [
                                                        html.Label(
                                                            "Ordenar categorias:", # Traduzido
                                                            id="reg-vs-col-categories-sort-label-"
                                                            + self.name,
                                                        ),
                                                        dbc.Tooltip(
                                                            "Como ordenar as categorias: Alfabeticamente, mais comuns "
                                                            "primeiro (Frequência), ou maior valor médio absoluto de SHAP primeiro (Impacto Shap)", # Traduzido
                                                            target="reg-vs-col-categories-sort-label-"
                                                            + self.name,
                                                        ),
                                                        dbc.Select(
                                                            id="reg-vs-col-categories-sort-"
                                                            + self.name,
                                                            options=[
                                                                {
                                                                    "label": "Alfabeticamente", # Traduzido
                                                                    "value": "alphabet",
                                                                },
                                                                {
                                                                    "label": "Frequência", # Traduzido
                                                                    "value": "freq",
                                                                },
                                                                {
                                                                    "label": "Impacto Shap", # Traduzido
                                                                    "value": "shap",
                                                                },
                                                            ],
                                                            value=self.cats_sort,
                                                        ),
                                                    ],
                                                    id="reg-vs-col-categories-sort-div-"
                                                    + self.name,
                                                ),
                                            ],
                                            md=4,
                                        ),
                                        hide=self.hide_cats_sort,
                                    ),
                                ]
                            )
                        ]
                    ),
                    hide=self.hide_footer,
                ),
            ]
        )

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        # Nota: As funções de plot podem precisar de ter os seus eixos traduzidos
        y_title_dict = {
            "observed": f"{self.explainer.target} Observado",
            "predicted": f"{self.explainer.target} Previsto",
            "difference": "Resíduo (Diferença)",
            "ratio": "Resíduo (Rácio)",
            "log-ratio": "Resíduo (Log-Rácio)"
        }
        y_title = y_title_dict.get(args["display"], args["display"].capitalize())
        x_title = args["col"] # Assume que o nome da coluna não precisa de tradução

        if args["display"] == "observed":
            fig = self.explainer.plot_y_vs_feature(
                args["col"],
                points=bool(args["points"]),
                winsor=args["winsor"],
                dropna=True,
                topx=args["cats_topx"],
                sort=args["cats_sort"],
                round=self.round,
                plot_sample=self.plot_sample,
            )
        elif args["display"] == "predicted":
            fig = self.explainer.plot_preds_vs_feature(
                args["col"],
                points=bool(args["points"]),
                winsor=args["winsor"],
                dropna=True,
                topx=args["cats_topx"],
                sort=args["cats_sort"],
                round=self.round,
                plot_sample=self.plot_sample,
            )
        else: # Residuals
            fig = self.explainer.plot_residuals_vs_feature(
                args["col"],
                residuals=args["display"], # Passa 'difference', 'ratio' ou 'log-ratio'
                points=bool(args["points"]),
                winsor=args["winsor"],
                dropna=True,
                topx=args["cats_topx"],
                sort=args["cats_sort"],
                round=self.round,
                plot_sample=self.plot_sample,
            )
        fig.update_layout(xaxis_title=x_title, yaxis_title=y_title, margin=dict(t=40, b=40, l=40, r=40))
        html = to_html.card(to_html.fig(fig), title=self.title, subtitle=self.subtitle)
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        @app.callback(
            [
                Output("reg-vs-col-graph-" + self.name, "figure"),
                Output("reg-vs-col-show-points-div-" + self.name, "style"),
                Output("reg-vs-col-n-categories-div-" + self.name, "style"),
                Output("reg-vs-col-categories-sort-div-" + self.name, "style"),
            ],
            [
                Input("reg-vs-col-col-" + self.name, "value"),
                Input("reg-vs-col-display-type-" + self.name, "value"),
                Input("reg-vs-col-show-points-" + self.name, "value"),
                Input("reg-vs-col-winsor-" + self.name, "value"),
                Input("reg-vs-col-n-categories-" + self.name, "value"),
                Input("reg-vs-col-categories-sort-" + self.name, "value"),
            ],
        )
        def update_residuals_graph(col, display, points, winsor, topx, sort):
            # Lógica interna, sem texto visível
            if (
                col in self.explainer.onehot_cols
                or col in self.explainer.categorical_cols # Assume que estes atributos existem
            ):
                style = {}
            else:
                style = dict(display="none")

            # Determinar títulos dos eixos
            y_title_dict = {
                "observed": f"{self.explainer.target} Observado",
                "predicted": f"{self.explainer.target} Previsto",
                "difference": "Resíduo (Diferença)",
                "ratio": "Resíduo (Rácio)",
                "log-ratio": "Resíduo (Log-Rácio)"
            }
            y_title = y_title_dict.get(display, display.capitalize())
            x_title = col # Assume que o nome da coluna não precisa de tradução

            if display == "observed":
                fig = self.explainer.plot_y_vs_feature(
                        col,
                        points=bool(points),
                        winsor=winsor,
                        dropna=True,
                        topx=topx,
                        sort=sort,
                        round=self.round,
                        plot_sample=self.plot_sample,
                    )
            elif display == "predicted":
                fig = self.explainer.plot_preds_vs_feature(
                        col,
                        points=bool(points),
                        winsor=winsor,
                        dropna=True,
                        topx=topx,
                        sort=sort,
                        round=self.round,
                        plot_sample=self.plot_sample,
                    )
            else: # Residuals
                fig = self.explainer.plot_residuals_vs_feature(
                        col,
                        residuals=display,
                        points=bool(points),
                        winsor=winsor,
                        dropna=True,
                        topx=topx,
                        sort=sort,
                        round=self.round,
                        plot_sample=self.plot_sample,
                    )

            fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
            return (fig, style, style, style)