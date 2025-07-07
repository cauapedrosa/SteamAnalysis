from dataclasses import dataclass

@dataclass(frozen=True)
class App:
    name: str
    id: str


apps = [
    App('Dota 2', '570'),
    App('Counter-Strike 2', '730'),
    App('Rust', '252490'),
    App('PUBG: BATTLEGROUNDS', '578080'),
    App('New World: Aeternum', '1063730'),
    App("Baldur's Gate 3", '1086940'),
    App('Apex Legends', '1172470'),
    App('ELDEN RING', '1245620'),
    App('Palworld', '1599340'),
    App('Black Myth: Wukong', '2358720'),
]

apps_by_name = {app.name: app for app in apps}
apps_by_id = {app.id: app for app in apps}

def get_app_by_name(name: str) -> App:
    return apps_by_name[name]

def get_app_by_id(app_id: str) -> App:
    return apps_by_id[app_id]
