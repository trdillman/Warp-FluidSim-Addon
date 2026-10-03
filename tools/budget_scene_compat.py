"""Storage adapter for frozen bridge scenes on the pre-split D7 reference."""


def restore_legacy_domain(owner, authored):
    """Copy authored Domain values to legacy Scene RNA without changing values."""
    for name in authored:
        if name == 'schema_version':
            continue
        prop = owner.bl_rna.properties.get(name)
        if prop is None:
            raise ValueError(f'legacy reference cannot read authored domain setting {name}')
        value = authored[name]
        if prop.type == 'ENUM' and isinstance(value, int):
            value = next(item.identifier for item in prop.enum_items if item.value == value)
        setattr(owner, name, value)
