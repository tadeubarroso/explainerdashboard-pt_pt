# -*- coding: utf-8 -*-

__all__ = [
    "delegates_kwargs",
    "delegates_doc",
    "update_params",
    "update_kwargs",
    "DummyComponent",
    "ExplainerComponent",
    "PosLabelSelector",
    "GraphPopout",
    "IndexSelector",
    "make_hideable",
    "get_dbc_tooltips",
    "encode_callables",
    "decode_callables",
    "reset_id_generator",
    "yield_id",
    "get_local_ip_adress",
    "instantiate_component",
]

import sys
from abc import ABC
import inspect
import types
from typing import Union, List, Tuple
from pathlib import Path
from importlib import import_module
import socket

import dash
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc

from . import to_html


# Stolen from https://www.fast.ai/2019/08/06/delegation/
# then extended to deal with multiple inheritance
def delegates_kwargs(to=None, keep=False):
    "Decorator: replace `**kwargs` in signature with params from `to`"

    def _f(f):
        from_f = f.__init__ if to is None else f
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)
        k = sigd.pop("kwargs")
        if to is None:
            for base_cls in f.__bases__:
                to_f = base_cls.__init__
                s2 = {
                    k: v
                    for k, v in inspect.signature(to_f).parameters.items()
                    if v.default != inspect.Parameter.empty and k not in sigd
                }
                sigd.update(s2)
        else:
            to_f = to
            s2 = {
                k: v
                for k, v in inspect.signature(to_f).parameters.items()
                if v.default != inspect.Parameter.empty and k not in sigd
            }
            sigd.update(s2)
        if keep:
            sigd["kwargs"] = k
        from_f.__signature__ = sig.replace(parameters=sigd.values())
        return f

    return _f


def delegates_doc(to=None, keep=False):
    "Decorator: replace `__doc__` with `__doc__` from `to`"

    def _f(f):
        from_f = f.__init__ if to is None else f
        if to is None:
            for base_cls in f.__bases__:
                to_f = base_cls.__init__
        else:
            if isinstance(to, types.FunctionType):
                to_f = to
            else:
                to_f = to.__init__
        from_f.__doc__ = to_f.__doc__
        return f

    return _f


def update_params(kwargs, **params):
    """kwargs override params"""
    return dict(params, **kwargs)


def update_kwargs(kwargs, **params):
    """params override kwargs"""
    return dict(kwargs, **params)


def encode_callables(obj):
    """replaces all callables (functions) in obj with a dict specifying module and name

    Works recursively through sub-list and sub-dicts"""
    if callable(obj):
        return dict(__callable__=dict(module=obj.__module__, name=obj.__name__))
    if isinstance(obj, dict):
        return {k: encode_callables(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [encode_callables(o) for o in obj]
    return obj


def decode_callables(obj):
    """replaces all dict-encoded callables in obj with the appropriate function

    Works recursively through sub-list and sub-dicts"""
    if isinstance(obj, dict) and "__callable__" in obj:
        return getattr(
            import_module(obj["__callable__"]["module"]), obj["__callable__"]["name"]
        )
    if isinstance(obj, dict):
        return {k: decode_callables(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decode_callables(o) for o in obj]
    return obj


def id_generator(prefix="id", start=0):
    """generator that generatores unique consecutive id's starting with 'id' + number

    Can be reset with reset_id_generator()"""
    i = start
    while True:
        yield prefix + str(i), i
        i += 1


def reset_id_generator(prefix="id", start=0):
    """resets the global id generator"""
    global id_gen
    id_gen = id_generator(prefix, start)


def yield_id(return_i=False):
    """yields the next unique consecutive id. Reset using reset_id_generator()"""
    global id_gen
    str_id, i = next(id_gen)
    if return_i:
        return str_id, i
    return str_id


reset_id_generator()


def get_local_ip_adress():
    """returns the local ip adress"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(("10.255.255.255", 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = "127.0.0.1"
    finally:
        s.close()
    return IP


def get_dbc_tooltips(dbc_table, desc_dict, hover_id, name):
    """Return a dbc.Table and a list of dbc.Tooltips.

    Args:
        dbc_table (dbc.Table): Table with first column consisting of label
        desc_dict (dict): dict that map labels to a description (str)
        hover_id (str): dash component_id base: tooltips will have
            component_id=f"{hover_id}-{label}-{name}"
        name (str): name to be used in hover_id

    Returns:
        dbc.Table, List[dbc.Tooltip]
    """
    tooltips_dict = {}
    for tr in dbc_table.children[1].children:
        tds = tr.children
        label = tds[0].children
        if label in desc_dict:
            tr.id = f"{hover_id}-{label}-" + name
            tooltips_dict[label] = desc_dict[label]

    tooltips = [
        dbc.Tooltip(desc, target=f"{hover_id}-{label}-" + name, placement="top")
        for label, desc in tooltips_dict.items()
    ]

    return dbc_table, tooltips


def make_hideable(element, hide=False):
    """helper function to optionally not display an element in a layout.

    This is used for all the hide_ flags in ExplainerComponent constructors.
    e.g. hide_cutoff=True to hide a cutoff slider from a layout:

    Example:
        make_hideable(dbc.Col([cutoff.layout()]), hide=hide_cutoff)

    Args:
        hide(bool): wrap the element inside a hidden html.div. If the element
                    is a dbc.Col or a dbc.Row, wrap element.children in
                    a hidden html.Div instead. Defaults to False.
    """
    if hide:
        if isinstance(element, dbc.Col) or isinstance(element, dbc.Row):
            return html.Div(element.children, style=dict(display="none"))
        else:
            return html.Div(element, style=dict(display="none"))
    else:
        return element


class DummyComponent:
    def __init__(self):
        pass

    def layout(self):
        return None

    def register_callbacks(self, app):
        pass


class ExplainerComponent(ABC):
    """ExplainerComponent is a bundle of a dash layout and callbacks that
    make use of an Explainer object.

    An ExplainerComponent can have ExplainerComponent subcomponents, that
    you register with register_components(). If the component depends on
    certain lazily calculated Explainer properties, you can register these
    with register_dependencies().

    ExplainerComponent makes sure that:

    1. Callbacks of subcomponents are registered.
    2. Lazily calculated dependencies (even of subcomponents) can be calculated.
    3. Pos labels selector id's of all subcomponents can be calculated.

    Each ExplainerComponent adds a unique uuid name string to all elements, so
    that there is never a name clash even with multiple ExplanerComponents of
    the same type in a layout.

    Important:
        define your callbacks in component_callbacks() and
        ExplainerComponent will register callbacks of subcomponents in addition
        to component_callbacks() when calling register_callbacks()
    """

    _state_props = {}

    def __init__(self, explainer, title=None, name=None):
        """initialize the ExplainerComponent

        Args:
            explainer (Explainer): explainer object constructed with e.g.
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to None.
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
        """
        self._store_child_params(no_param=["explainer"])
        if not hasattr(self, "name") or self.name is None:
            self.name = name or yield_id()
        if not hasattr(self, "title") or self.title is None:
            # Traduzido o título padrão
            self.title = title or "Personalizado"

        self._components = []
        self._dependencies = []

    def _store_child_params(self, no_store=None, no_attr=None, no_param=None):
        if not hasattr(self, "_stored_params"):
            self._stored_params = {}
        child_frame = sys._getframe(2)
        child_args = child_frame.f_code.co_varnames[1 : child_frame.f_code.co_argcount]
        child_dict = {arg: child_frame.f_locals[arg] for arg in child_args}

        if isinstance(no_store, bool) and no_store:
            return
        else:
            if no_store is None:
                no_store = tuple()

        if isinstance(no_attr, bool) and no_attr:
            dont_attr = True
        else:
            if no_attr is None:
                no_attr = tuple()
            dont_attr = False

        if isinstance(no_param, bool) and no_param:
            dont_param = True
        else:
            if no_param is None:
                no_param = tuple()
            dont_param = False

        for name, value in child_dict.items():
            if not dont_attr and name not in no_store and name not in no_attr:
                setattr(self, name, value)
            if not dont_param and name not in no_store and name not in no_param:
                self._stored_params[name] = value

        self._stored_params = encode_callables(self._stored_params)

    def exclude_callbacks(self, *components):
        """exclude certain subcomponents from the register_components scan"""
        if not hasattr(self, "_exclude_components"):
            self._exclude_components = []
        for comp in components:
            if (
                isinstance(comp, ExplainerComponent)
                and comp not in self._exclude_components
            ):
                self._exclude_components.append(comp)

    def register_components(self, *components):
        """register subcomponents so that their callbacks will be registered
        and dependencies can be tracked

        Args:
            scan_self (bool, optional): scan self.__dict__ and add all
            ExplainerComponent attributes to _components. Defaults to True
        """
        if not hasattr(self, "_components"):
            self._components = []
        if not hasattr(self, "_exclude_components"):
            self._exclude_components = []
        for comp in components:
            if (
                isinstance(comp, ExplainerComponent)
                and comp not in self._components
                and comp not in self._exclude_components
            ):
                self._components.append(comp)
            elif hasattr(comp, "__iter__"):
                for subcomp in comp:
                    if (
                        isinstance(subcomp, ExplainerComponent)
                        and subcomp not in self._components
                        and subcomp not in self._exclude_components
                    ):
                        self._components.append(subcomp)
                    else:
                        print(
                            # Traduzido
                            f"{subcomp.__class__.__name__} não é um ExplainerComponent, não será adicionado a self.components"
                        )
            # else: # Comentado pois pode ser muito verboso
            #     print(
            #         f"{comp.__class__.__name__} is not an ExplainerComponent so not adding to self.components"
            #     )

        for k, v in self.__dict__.items():
            if (
                k != "_components"
                and isinstance(v, ExplainerComponent)
                and v not in self._components
                and v not in self._exclude_components
            ):
                self._components.append(v)

    def has_pos_label_connector(self):
        if not hasattr(self, "_components"):
            self._components = []
        for comp in self._components:
            if str(type(comp)).endswith("PosLabelConnector'>"):
                return True
            elif hasattr(comp, 'has_pos_label_connector') and comp.has_pos_label_connector(): # Adicionado hasattr
                return True
        return False

    def register_dependencies(self, *dependencies):
        """register dependencies: lazily calculated explainer properties that
        you want to calculate *before* starting the dashboard"""
        for dep in dependencies:
            if isinstance(dep, str):
                self._dependencies.append(dep)
            elif hasattr(dep, "__iter__"):
                for subdep in dep:
                    if isinstance(subdep, str):
                        self._dependencies.append(subdep)
                    else:
                        print(
                            # Traduzido
                            f"{subdep.__class__.__name__} não é uma string, não será adicionado a self.dependencies"
                        )
            else:
                print(
                    # Traduzido
                    f"{dep.__class__.__name__} não é uma string ou lista de strings, não será adicionado a self.dependencies"
                )

    @property
    def dependencies(self):
        """returns a list of unique dependencies of the component
        and all subcomponents"""
        if not hasattr(self, "_dependencies"):
            self._dependencies = []
        self.register_components()
        deps = self._dependencies
        for comp in self._components:
            deps.extend(comp.dependencies)
        deps = list(set(deps))
        return deps

    @property
    def component_imports(self):
        """returns a list of ComponentImport namedtuples("component", "module")
        all components and and subcomponents"""
        self.register_components()
        _component_imports = [(self.__class__.__name__, self.__class__.__module__)]
        for comp in self._components:
            _component_imports.extend(comp.component_imports)
        return list(set(_component_imports))

    def get_state_tuples(self):
        """returns a list of State (id, property) tuples for the component and all subcomponents"""
        self.register_components()
        for key, tup in self._state_props.items():
            if (
                not isinstance(tup, tuple)
                or len(tup) != 2
                or not isinstance(tup[0], str)
                or not isinstance(tup[1], str)
            ):
                raise ValueError(
                    # Traduzido
                    f"Esperado uma tupla (id:str, property:str) mas {self} tem _state_props['{key}'] == {tup}"
                )
        _state_tuples = [
            (id_ + self.name, prop_) for id_, prop_ in self._state_props.values()
        ]
        for comp in self._components:
            _state_tuples.extend(comp.get_state_tuples())
        return sorted(list(set(_state_tuples)))

    def get_state_args(self, state_dict=None):
        """returns _state_dict with correct self.name attached
        if state_dict is passed then replace the state_id_prop_tuples with their values
        from state_dict or else as a property.
        """
        state_tuples = {
            k: (v[0] + self.name, v[1]) for k, v in self._state_props.items()
        }
        state_args = {}
        state_dict = state_dict or {}
        for param, (id_, prop_) in state_tuples.items():
            if (id_, prop_) in state_dict:
                state_args[param] = state_dict[(id_, prop_)]
            elif hasattr(self, param):
                state_args[param] = getattr(self, param)
        return state_args

    @property
    def pos_labels(self):
        """returns a list of unique pos label selector elements
        of the component and all subcomponents"""

        self.register_components()
        pos_labels = []
        for k, v in self.__dict__.items():
            if isinstance(v, PosLabelSelector) and v.name not in pos_labels:
                pos_labels.append("pos-label-" + v.name)
        # if hasattr(self, 'selector') and isinstance(self.selector, PosLabelSelector):
        #     pos_labels.append('pos-label-'+self.selector.name)
        for comp in self._components:
            pos_labels.extend(comp.pos_labels)
        pos_labels = list(set(pos_labels))
        return pos_labels

    def calculate_dependencies(self):
        """calls all properties in self.dependencies so that they get calculated
        up front. This is useful to do before starting a dashboard, so you don't
        compute properties multiple times in parallel."""
        for dep in self.dependencies:
            try:
                attribute = getattr(self.explainer, dep)
                if callable(attribute):
                    _ = attribute()
            except:
                # Traduzido
                ValueError(
                    f"Falha ao gerar dependência '{dep}': "
                    f"Falha ao calcular ou recuperar a propriedade do explainer explainer.{dep}..."
                )

    def layout(self):
        """layout to be defined by the particular ExplainerComponent instance.
        All element id's should append +self.name to make sure they are unique."""
        return None

    def to_html(self, state_dict: dict = None, add_header: bool = True):
        """return static html for this component and all subcomponents.

        Args:
            state_dict (dict): dictionary with id_prop_tuple as keys and state as value.
        """
        html = to_html.div("")
        if add_header:
            return to_html.add_header(html)
        return html

    def save_html(self, filename: Union[str, Path]):  # noqa: E821
        """Store output of to_html to a file

        Args:
            filename (str, Path): filename to store html
        """
        html = self.to_html(add_header=True)
        with open(filename, "w") as file:
            file.write(html)

    def component_callbacks(self, app):
        """register callbacks specific to this ExplainerComponent."""
        if hasattr(self, "_register_callbacks"):
            print(
                # Traduzido
                "Aviso: o uso de _register_callbacks() será descontinuado!"
                " Use component_callbacks() a partir de agora..."
            )
            self._register_callbacks(app)

    def register_callbacks(self, app):
        """First register callbacks of all subcomponents, then call
        component_callbacks(app)
        """
        self.register_components()
        for comp in self._components:
            comp.register_callbacks(app)
        self.component_callbacks(app)


class PosLabelSelector(ExplainerComponent):
    """For classifier models displays a drop down menu with labels to be selected
    as the positive class.
    """

    def __init__(
        self, explainer, title="Seletor de Classe Positiva", name=None, pos_label=None # Traduzido title default
    ):
        """Generates a positive label selector with element id 'pos_label-'+self.name

        Args:
            explainer (Explainer): explainer object constructed with e.g.
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to None.
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to 'Seletor de Classe Positiva'. # Atualizado docstring
            pos_label (int, optional): Initial pos label. Defaults to
                        explainer.pos_label.
        """
        super().__init__(explainer, title, name)
        if pos_label is not None:
            self.pos_label = explainer.pos_label_index(pos_label)
        else:
            self.pos_label = explainer.pos_label

    def layout(self):
        if self.explainer.is_classifier:
            return html.Div(
                [
                    html.Div(
                        [
                            dbc.Label(
                                "Classe positiva:", # Traduzido
                                html_for="pos-label-" + self.name,
                                id="pos-label-label-" + self.name,
                                style={"font-size": "16"},
                            ),
                            dbc.Tooltip(
                                # Traduzido
                                "Selecione a etiqueta a ser definida como classe positiva",
                                target="pos-label-label-" + self.name,
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="pos-label-" + self.name,
                                options=[
                                    {"label": label, "value": i}
                                    for i, label in enumerate(self.explainer.labels)
                                ],
                                value=self.pos_label,
                                optionHeight=24,
                                clearable=False,
                                style={
                                    "height": "24",
                                    "font-size": "24",
                                },
                            )
                        ]
                    ),
                ]
            )
        else:
            return html.Div(
                [dcc.Input(id="pos-label-" + self.name)], style=dict(display="none")
            )


class IndexSelector(ExplainerComponent):
    """Either shows a dropdown or a free text input field for selecting an index"""

    def __init__(
        self,
        explainer,
        name: str = None,
        index: str = None,
        index_dropdown: bool = True,
        max_idxs_in_dropdown: int = 1000,
        **kwargs,
    ):
        """generates an index selector, either (dynamic) dropdown or free text field with input checker

        Args:
            explainer (BaseExplainer): explainer
            name (str, optional): dash id to assign to the component. Defaults to None, in which case a unique identifier gets generated.
            index (str, optional): initial index to select and display. Defaults to None.
            index_dropdown (bool, optional): if set to false, input is an open text input instead of a dropdown. Defaults to True.
            max_idxs_in_dropdown (int, optional): If the number of rows (idxs) in X_test is larger than this,
                use a servers-side dynamically updating set of dropdown options instead of storing all index
                    options client side. Defaults to 1000.
        """
        super().__init__(explainer, name=name)
        # Assume explainer.index_name já está traduzido (foi feito no ficheiro explainers.py)
        self.index_name = explainer.index_name
        # Correção: o name do componente deve ser o name passado ou gerado, não o index_name
        # self.name já é definido em super().__init__

    def layout(self):
        if self.index_dropdown:
            index_list = self.explainer.get_index_list()
            if len(index_list) > self.max_idxs_in_dropdown:
                return dcc.Dropdown(
                    id=self.name,
                    # Traduzido placeholder
                    placeholder=f"Pesquisar {self.index_name} aqui...",
                    value=self.index,
                    searchable=True,
                )
            else:
                return dcc.Dropdown(
                    id=self.name,
                     # Traduzido placeholder
                    placeholder=f"Selecionar {self.index_name} aqui...",
                    options=index_list.astype(str).to_list(),
                    searchable=True,
                    value=self.index,
                )
        else:
            return dbc.Input(
                id=self.name,
                # Traduzido placeholder
                placeholder=f"Digite {self.index_name} aqui...",
                value=self.index,
                debounce=True,
                type="text",
            )

    def component_callbacks(self, app):
        if self.index_dropdown:
            if len(self.explainer.get_index_list()) > self.max_idxs_in_dropdown:

                @app.callback(
                    Output(self.name, "options"),
                    Input(self.name, "search_value"),
                    Input(self.name, "value"),
                )
                def update_options(search_value, index):
                    trigger_props = [
                        trigger["prop_id"].split(".")[-1]
                        for trigger in dash.callback_context.triggered
                    ]
                    if "value" in trigger_props or not search_value:
                        new_options = [index] if index is not None else []
                    else:
                        new_options = [
                            idx
                            for idx in self.explainer.get_index_list()
                            if (str(search_value) in idx) or (idx == str(index))
                        ]
                    return new_options[: self.max_idxs_in_dropdown]

        else:

            @app.callback(
                [Output(self.name, "valid"), Output(self.name, "invalid")],
                [Input(self.name, "value")],
            )
            def update_valid_index(index):
                if index is not None:
                    if self.explainer.index_exists(index):
                        return True, False
                    else:
                        return False, True
                return False, False


class GraphPopout(ExplainerComponent):
    """Provides a way to open a modal popup with the content of a graph figure."""

    def __init__(
        self,
        name: str,
        graph_id: str,
        title: str = "Expandir", # Traduzido default
        description: str = None,
        button_text: str = "Expandir", # Traduzido default
        button_size: str = "sm",
        button_outline: bool = True,
    ):
        """
        Args:
            name (str): name id for this GraphPopout. Should be unique.
            graph_id (str): id of of the dcc.Graph component that gets included
                in the modal.
            title (str): Title above the modal. Defaults to Expandir. # Atualizado docstring
            description (str): description of the graph to be include in the footer.
            button_text (str, optional): Text on the Button. Defaults to "Expandir". # Atualizado docstring
            button_size (str, optiona). Size of the button.Defaults to "sm" or small.
            button_outline (bool, optional). Show outline of button instead with fill color.
                Defaults to True.
        """

        self.title = title
        self.name = name
        self.graph_id = graph_id
        self.description = description
        self.button_text, self.button_size, self.button_outline = (
            button_text,
            button_size,
            button_outline,
        )

    def layout(self):
        return html.Div(
            [
                dbc.Button(
                    self.button_text, # Usa o texto do botão (já traduzido no init ou passado)
                    id=self.name + "modal-open",
                    size=self.button_size,
                    color="secondary",
                    outline=self.button_outline,
                ),
                dbc.Modal(
                    [
                        # ToDo the X on the top right is not rendered properly, disabling
                        dbc.ModalHeader(dbc.ModalTitle(self.title), close_button=True), # Usa o título (já traduzido no init ou passado)
                        dbc.ModalBody(
                            dcc.Graph(
                                id=self.name + "-modal-graph",
                                style={"max-height": "none", "height": "80%"},
                            )
                        ),
                        dbc.ModalFooter(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        dbc.Button(
                                                            html.Small("Descrição"), # Traduzido
                                                            id=self.name
                                                            + "-show-description",
                                                            color="link",
                                                            className="text-muted ml-auto",
                                                        ),
                                                        dbc.Fade(
                                                            [
                                                                html.Small(
                                                                    self.description,
                                                                    className="text-muted",
                                                                )
                                                            ],
                                                            id=self.name + "-fade",
                                                            is_in=True,
                                                            appear=True,
                                                        ),
                                                    ],
                                                    style=dict(
                                                        display="none"
                                                        if not self.description
                                                        else None
                                                    ),
                                                )
                                            ],
                                            className="text-left",
                                        ),
                                        html.Div(
                                            [
                                                dbc.Button(
                                                    "Fechar", # Traduzido
                                                    id=self.name + "-modal-close",
                                                    className="mr-auto",
                                                )
                                            ],
                                            className="text-right",
                                            style=dict(float="right"),
                                        ),
                                    ],
                                    style={"display": "flex"},
                                ),
                            ],
                            className="justify-content-between",
                        ),
                    ],
                    id=self.name + "-modal",
                    size="xl",
                ),
            ],
            style={"display": "flex", "justify-content": "flex-end"},
        )

    def component_callbacks(self, app):
        @app.callback(
            [
                Output(self.name + "-modal", "is_open"),
                Output(self.name + "-modal-graph", "figure"),
            ],
            [
                Input(self.name + "modal-open", "n_clicks"),
                Input(self.name + "-modal-close", "n_clicks"),
            ],
            [State(self.name + "-modal", "is_open"), State(self.graph_id, "figure")],
        )
        def toggle_modal(open_modal, close_modal, modal_is_open, fig):
            if open_modal or close_modal:
                ctx = dash.callback_context
                button_id = ctx.triggered[0]["prop_id"].split(".")[0]
                if button_id == self.name + "modal-open":
                    return (not modal_is_open, fig)
                else:
                    return (not modal_is_open, dash.no_update)
            return (modal_is_open, dash.no_update)

        if self.description is not None:

            @app.callback(
                Output(self.name + "-fade", "is_in"),
                [Input(self.name + "-show-description", "n_clicks")],
                [State(self.name + "-fade", "is_in")],
            )
            def toggle_fade(n_clicks, is_in):
                if not n_clicks:
                    # Button has never been clicked
                    return False
                return not is_in


def instantiate_component(component, explainer, name=None, **kwargs):
    """Returns an instantiated ExplainerComponent.
    If the component input is just a class definition, instantiate it with
    explainer and k**wargs.
    If it is already an ExplainerComponent instance then return it.
    If it is any other instance with layout and register_components methods,
    then add a name property and return it.

    Args:
        component ([type]): Either a class definition or instance
        explainer ([type]): An Explainer object that will be used to instantiate class definitions
        name (str): name to assign to ExplainerComponent
        kwargs: kwargs will be passed on to the instance

    Raises:
        ValueError: if component is not a subclass or instance of ExplainerComponent,
                or is an instance without layout and register_callbacks methods

    Returns:
        ExplainerComponent: instantiated component
    """

    if inspect.isclass(component) and issubclass(component, ExplainerComponent):
        init_argspec = inspect.getfullargspec(component.__init__)
        assert len(init_argspec.args) > 1 and init_argspec.args[1] == "explainer", (
            # Traduzido
            f"O primeiro parâmetro de {component.__name__}.__init__ deve ser 'explainer'. "
            f"Em vez disso, o __init__ é: {component.__name__}.__init__{inspect.signature(component.__init__)}"
        )
        if not init_argspec.varkw:
            kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in init_argspec.args + init_argspec.kwonlyargs
            }
        if "name" in init_argspec.args + init_argspec.kwonlyargs:
            component = component(explainer, name=name, **kwargs)
        else:
            print(
                 # Traduzido
                f"ExplainerComponent {component.__name__} não aceita um parâmetro name, "
                f"portanto não é possível atribuir name='{name}': "
                f"{component.__name__}.__init__{inspect.signature(component.__init__)}. "
                "Certifique-se de definir super().__init__(name=...) explicitamente "
                "dentro do __init__ se quiser implementar em vários workers ou num cluster, "
                "caso contrário, cada instância no cluster gerará o seu próprio nome uuid aleatório!"
            )
            component = component(explainer, **kwargs)
        return component
    elif isinstance(component, ExplainerComponent):
        return component
    else:
        raise ValueError(f"{component} não é um ExplainerComponent válido...") # Traduzido