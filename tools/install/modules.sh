# Optional source modules for the head's local expert worker or a node.
# Compile and check them in the private build before replacing a working binary.
# User-owned module sources live outside src/, so uninstall keeps their work.

prepare_backend_modules() {
    local source manifest module choice
    custom_modules=()
    if ((${#module_paths[@]} == 0 && !yes)); then
        choice=$(ask "Expert backends: use the included choices or add a local source module?" included "included|add-module")
        if [[ $choice == add-module ]]; then
            module_paths+=("$(ask 'Backend module folder (containing backend.json)' '')")
        fi
    fi
    for manifest in "$PREFIX/modules/"*/backend.json; do
        [[ -f $manifest ]] || continue
        source=${manifest%/backend.json}
        module=$(python3 "$SRC_TREE/tools/backend-module.py" validate "$source") || die "invalid saved backend module: $source"
        # The installed manager includes its previous compiled source copy.
        # A downloaded manager must not silently replace a newly bundled module.
        [[ ! -d $stage/backends/modules/$module || $SRC_TREE -ef $WORK ]] || die "saved module $module collides with a bundled module; give your local module a unique name"
        python3 "$SRC_TREE/tools/backend-module.py" copy "$source" "$stage/backends/modules" --replace >/dev/null || die "could not stage $module"
        rm -rf "$stage/backends/modules/.$module.previous"
        custom_modules+=("$module")
    done
    for source in ${module_paths[@]+"${module_paths[@]}"}; do
        module=$(python3 "$SRC_TREE/tools/backend-module.py" validate "$source") || die "choose a valid source module folder"
        if [[ -d $stage/backends/modules/$module && " ${custom_modules[*]-} " != *" $module "* ]]; then
            die "module $module collides with a bundled module; choose a unique name"
        fi
        python3 "$SRC_TREE/tools/backend-module.py" copy "$source" "$stage/backends/modules" --replace >/dev/null || die "could not stage $module"
        rm -rf "$stage/backends/modules/.$module.previous"
        [[ " ${custom_modules[*]-} " == *" $module "* ]] || custom_modules+=("$module")
        say "Added source module $module. It will appear among the backends if this machine supports it."
    done
}

check_backend_modules() {
    local listing module
    ((${#custom_modules[@]})) || [[ -n $requested_backend ]] || return 0
    listing=$(cd "$stage" && ./fronk backends --list) || die "could not inspect the staged backends"
    for module in ${custom_modules[@]+"${custom_modules[@]}"}; do
        if awk -v name="$module" '$2==name && $3=="runs" && $4=="here" {ok=1} END {exit !ok}' <<< "$listing"; then
            say "Checking backend module $module before installation."
            (cd "$stage" && ./fronk backends --check "$module") > "$STATE/backend-$module.log" 2>&1 ||
                die "module $module failed SELFTEST; the previous binary is preserved. See $STATE/backend-$module.log"
        else
            say "Module $module is not available on this platform/hardware; it will not be offered for selection."
        fi
    done
    if [[ -n $requested_backend && $requested_backend != auto ]]; then
        awk -v name="$requested_backend" '$2==name && $3=="runs" && $4=="here" {ok=1} END {exit !ok}' <<< "$listing" ||
            die "requested backend $requested_backend cannot run here; inspect the available choices with fronk backends --list"
    fi
}

save_backend_modules() {
    local module
    for module in ${custom_modules[@]+"${custom_modules[@]}"}; do
        python3 "$SRC_TREE/tools/backend-module.py" copy "$stage/backends/modules/$module" "$PREFIX/modules" --replace >/dev/null ||
            die "could not preserve the source of module $module"
    done
}

remove_old_module_copies() {
    # The editable originals live in PREFIX/modules. Drop only recorded build
    # inputs absent from the new package, so a removed module cannot return on
    # the next upgrade through the installed manager.
    local file
    [[ -f $STATE/install-files ]] || return 0
    while IFS= read -r file; do
        [[ $file == ./backends/modules/* && $file != */../* ]] || continue
        [[ -e $stage/$file ]] || rm -f "$WORK/$file"
    done < "$STATE/install-files"
    if [[ -d $WORK/backends/modules ]]; then find "$WORK/backends/modules" -depth -type d -empty -delete; fi
}
