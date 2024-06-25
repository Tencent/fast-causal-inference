from enum import Enum
from typing import List, Dict


class OlapEngineType(Enum):
    CLICKHOUSE = 1
    STARROCKS = 2

    def __hash__(self):
        return self.value

    def __str__(self):
        return self.name.lower()


class DfContext:
    def __init__(self, engine: OlapEngineType, dataframe):
        self._engine = engine
        self._dataframe = dataframe

    @property
    def engine(self) -> OlapEngineType:
        return self._engine

    @property
    def dataframe(self):
        assert self._dataframe is not None
        return self._dataframe

    @dataframe.setter
    def dataframe(self, df):
        self._dataframe = df


class DfColumnNode:
    def __init__(self, alias: str = None, type: str = "Float64"):
        self._alias = alias
        self._type = type
        self._cache = None
        self._is_agg = False

    def sql(self, ctx: DfContext, show_alias=False) -> str:
        raise Exception("Not implemented.")
    
    @property
    def alias(self):
        return self._alias

    @alias.setter
    def alias(self, alias_):
        self._alias = alias_

    @property
    def type(self):
        return self._type
    
    @property
    def cache(self):
        return self._cache

    @cache.setter
    def cache(self, col):
        self._cache = col

    @type.setter
    def type(self, type_):
        self._type = type_

    @property
    def is_agg(self):
        return self._is_agg

    @is_agg.setter
    def is_agg(self, is_agg_):
        self._is_agg = is_agg_

    def find_column(self, alias: str):
        if self.alias == alias:
            return self


class FnArg:
    def __init__(self, *, name=None, is_param=False, default=None, is_variadic=False):
        self._name = name
        self._is_param = is_param
        self._default = default
        self._is_variadic = is_variadic
        self._column = None

    @property
    def name(self):
        return self._name

    @property
    def is_param(self):
        return self._is_param

    @property
    def default(self):
        return self._default

    @property
    def is_variadic(self):
        return self._is_variadic

    @property
    def column(self) -> DfColumnNode:
        return self._column

    @column.setter
    def column(self, col: DfColumnNode):
        self._column = col

    def sql(self, ctx):
        return self._column.sql(ctx)


class DfFunction:
    fn_names = dict()

    @property
    def alias(self):
        return self._alias

    @alias.setter
    def alias(self, alias_):
        self._alias = alias_

    @classmethod
    def is_agg_func(cls):
        return False

    def __hash__(self):
        return hash((self.__class__, self._alias))

    def fn_name(self, ctx: DfContext) -> str:
        signature = (type(self).__name__, ctx.engine)
        name = DfFunction.fn_names.get(signature)
        if name is None:
            raise Exception(
                f"function signature({type(self).__name__}, {ctx.engine}) not found."
            )
        return name

    def sql(self, params: Dict, df_columns: List[DfColumnNode], ctx: DfContext) -> str:
        real_args, real_params, arg_dict = self.fill_args(params, df_columns, ctx)
        impl = getattr(self, f"sql_impl_{ctx.engine}", None)
        if impl is None:
            raise Exception(
                f"fail to find sql impl method: {ctx.engine}, try sql_impl_{ctx.engine}."
            )
        return impl(ctx, real_args, real_params, arg_dict)

    def sql_impl_default(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        result = self.fn_name(ctx)

        def expand(arg):
            if isinstance(arg.column, list):
                return ", ".join(map(lambda p: p.sql(ctx), arg.column))
            return arg.column.sql(ctx)

        if fn_params:
            result += f"({', '.join(map(expand, fn_params))})"
        if fn_args:
            result += f"({', '.join(map(expand, fn_args))})"
        if not fn_params and not fn_args:
            result += "()"
        return result

    def sql_impl_clickhouse(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        return self.sql_impl_default(ctx, fn_args, fn_params, arg_dict)

    def sql_impl_starrocks(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        return self.sql_impl_default(ctx, fn_args, fn_params, arg_dict)

    def fill_args(self, params: Dict, df_columns: List[DfColumnNode], ctx: DfContext):
        for arg in self._args:
            if not isinstance(arg, FnArg):
                raise Exception(f"Argument({arg}) is not a FnArg")
        fn_params, fn_args = [], []
        is_variadic = False
        for arg in self._args:
            if is_variadic:
                raise Exception("Only the last arg can be variadic.")
            if arg.is_param:
                fn_params.append(arg)
            else:
                fn_args.append(arg)
            if arg.is_variadic:
                is_variadic = True
        if not is_variadic and len(df_columns) > len(self._args):
            raise Exception(
                f"num args({len(df_columns)}) of function({self.fn_name(ctx)}) "
                f"is larger than expected num args({len(self._args)}."
            )
        if len(params) > len(fn_params):
            raise Exception(
                f"num params({len(params)}) of function({self.fn_name(ctx)}) "
                f"is larger than expected num params({len(fn_params)}."
            )
        real_args, real_params = [], []
        arg_dict = {}
        for i in range(len(fn_args)):
            if i >= len(df_columns):
                if fn_args[i].default is None:
                    raise Exception(f"Arg `{fn_args[i].name}` is not given.")
                col = DfColumnLeafNode(fn_args[i].default)
            else:
                col = df_columns[i]
            fn_args[i].column = col
            real_args.append(fn_args[i])
            arg_dict[fn_args[i].name] = fn_args[i]
        if is_variadic:
            real_args[-1].column = [real_args[-1].column]
            for i in range(len(fn_args), len(df_columns)):
                real_args[-1].column.append(df_columns[i])
            arg_dict[fn_args[-1].name] = real_args[-1]
        for param in fn_params:
            real_params.append(param)
            if param.name not in params:
                if param.default is None:
                    raise Exception(
                        f"missing param({param}) of function({self.fn_name(ctx)}"
                    )
                param.column = DfColumnLeafNode(str(param.default))
                continue
            param.column = DfColumnLeafNode(str(params[param.name]))
            arg_dict[param.name] = param
        return real_args, real_params, arg_dict


def register_fn(*, engine: OlapEngineType, name: str):
    def decorator(cls):
        if engine is None or name is None:
            raise Exception(f"engine({engine}) and name({name}) can not be None.")
        # if DfFunction.fn_names.get((cls.__name__, engine)) is not None:
        #     raise Exception(f"function signature({cls.__name__}, {engine}) already exists.")
        DfFunction.fn_names[(cls.__name__, engine)] = name
        return cls

    return decorator


def define_args(*args):
    def decorator(cls):
        def __init__(self):
            self._args = args
            self._alias = None

        cls.__init__ = __init__
        return cls

    return decorator


def aggregrate(cls):
    def is_agg_func(cls):
        return True

    cls.is_agg_func = is_agg_func
    return cls


class DfColumnLeafNode(DfColumnNode):
    def __init__(self, column_name: str, alias: str = None, type="Float64"):
        super(DfColumnLeafNode, self).__init__(alias, type)
        self._column_name = column_name

    def sql(self, ctx: DfContext, show_alias=False) -> str:
        if self.cache is None:
            self.cache = self._column_name + (
                f" as {self._alias}" if self._alias is not None and show_alias else ""
            )
        return self.cache

    def find_column(self, alias: str):
        if alias == "" or alias is None:
            return
        if self.alias == alias or self._column_name == alias:
            return self


class DfColumnInternalNode(DfColumnNode):
    def __init__(
        self,
        fn: DfFunction,
        params: Dict,
        columns: List[DfColumnNode],
        alias: str = None,
        type="Float64",
    ):
        super(DfColumnInternalNode, self).__init__(alias, type)
        self._columns = columns
        self._fn = fn
        self._params = params

    def sql(self, ctx: DfContext, show_alias=False) -> str:
        if self.cache is None:
            self.cache = self._fn.sql(self._params, self._columns, ctx) + (
                f" as {self._alias}" if self._alias is not None and show_alias else ""
            )
        return self.cache
        
    def find_column(self, alias: str):
        if self.alias == alias:
            return self
        for col in self._columns:
            tar = col.find_column(alias)
            if tar is not None:
                return tar

    @property
    def params(self):
        return self._params


def df_2_table(df):
    new_df = df.materializedView()
    return new_df.getTableName()


def table_2_df(table):
    return readClickHouse(table)
